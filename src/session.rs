//! Session management for Cursor ACP.
//!
//! This module handles individual sessions, spawning Cursor CLI processes,
//! and translating events to ACP session updates.
//!
//! Architecture:
//! - Messages are sent to a channel and processed sequentially
//! - A background task handles one CLI invocation at a time
//! - Prompt calls block until processing completes

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use crate::cursor_process::find_cursor_binary;
use agent_client_protocol::{
    Client, ClientCapabilities, ContentBlock, ContentChunk, Error, ImageContent, PromptRequest,
    SessionId, SessionNotification, SessionUpdate, StopReason, Terminal, TextContent, ToolCall,
    ToolCallContent, ToolCallId, ToolCallLocation, ToolCallStatus, ToolCallUpdate,
    ToolCallUpdateFields, ToolKind,
};
use tokio::sync::oneshot;
use tracing::{debug, error, info, warn};

use crate::cursor_process::{CursorProcess, CursorProcessParts};
use crate::stream_json::{StreamEvent, ToolCallSubtype};
use crate::ACP_CLIENT;
use tokio::sync::mpsc;

/// Build a single prompt string from ACP content blocks, preserving order.
/// Text blocks are concatenated; image blocks are embedded as markdown data URLs.
fn content_blocks_to_prompt(blocks: &[ContentBlock]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text(TextContent { text, .. }) => {
                if !text.is_empty() {
                    parts.push(text.clone());
                }
            }
            ContentBlock::Image(ImageContent {
                data, mime_type, ..
            }) => {
                parts.push(format!("![image](data:{};base64,{})", mime_type, data));
            }
            _ => {}
        }
    }
    parts.join("\n")
}

/// Messages that can be sent to the session's background task
enum SessionMessage {
    /// Process a prompt and return the result via the response channel
    Prompt {
        prompt_text: String,
        response_tx: oneshot::Sender<Result<StopReason, Error>>,
    },
    /// Cancel the current operation
    Cancel { response_tx: oneshot::Sender<()> },
    /// Set the model for future prompts
    SetModel { model: String },
    /// Set the mode for future prompts
    SetMode { mode: String },
}

/// A session represents an active conversation with Cursor
pub struct Session {
    /// Channel to send messages to the background task
    message_tx: mpsc::Sender<SessionMessage>,
    /// Notify the worker to cancel the current prompt (so it can react while blocked in process_events)
    cancel_tx: mpsc::Sender<()>,
    /// Cursor CLI's internal session ID for conversation continuity (shared with background task)
    cursor_session_id: Arc<Mutex<Option<String>>>,
}

impl Session {
    pub fn new(
        session_id: SessionId,
        cwd: PathBuf,
        client_capabilities: Arc<Mutex<ClientCapabilities>>,
    ) -> Self {
        let (message_tx, message_rx) = mpsc::channel(32);
        let (cancel_tx, cancel_rx) = mpsc::channel(4);
        let cursor_session_id = Arc::new(Mutex::new(None));

        // Spawn the background task that processes messages (cancel_rx passed in so worker
        // can react to cancel while blocked in process_events without double-borrow)
        let worker = SessionWorker::new(
            session_id.clone(),
            cwd,
            client_capabilities,
            cursor_session_id.clone(),
        );
        tokio::task::spawn_local(worker.run(message_rx, cancel_rx));

        Self {
            message_tx,
            cancel_tx,
            cursor_session_id,
        }
    }

    pub fn set_model(&self, model: String) {
        // Fire and forget - the worker will pick it up
        drop(self.message_tx.try_send(SessionMessage::SetModel { model }));
    }

    pub fn set_mode(&self, mode: String) {
        drop(self.message_tx.try_send(SessionMessage::SetMode { mode }));
    }

    pub fn set_cursor_session_id(&self, id: String) {
        *self.cursor_session_id.lock().unwrap() = Some(id);
    }

    pub fn cursor_session_id(&self) -> Option<String> {
        self.cursor_session_id.lock().unwrap().clone()
    }

    /// Send a prompt to the background worker and await the result.
    pub async fn prompt(&self, request: PromptRequest) -> Result<StopReason, Error> {
        let prompt_text = content_blocks_to_prompt(&request.prompt);

        if prompt_text.is_empty() {
            return Ok(StopReason::EndTurn);
        }

        let (response_tx, response_rx) = oneshot::channel();

        if self
            .message_tx
            .send(SessionMessage::Prompt {
                prompt_text,
                response_tx,
            })
            .await
            .is_err()
        {
            return Err(Error::internal_error().data("Session worker has stopped"));
        }

        // Wait for the worker to complete processing
        response_rx
            .await
            .map_err(|_| Error::internal_error().data("Session worker dropped response channel"))?
    }

    /// Cancel the current operation.
    /// Notifies the worker via cancel_tx so it can kill the child even while blocked in process_events;
    /// also sends a Cancel message so the worker acknowledges and the client can await.
    pub async fn cancel(&self) {
        // Notify worker immediately so it can react inside process_events (ignore if full/closed)
        let _ = self.cancel_tx.try_send(());
        let (response_tx, response_rx) = oneshot::channel();
        if self
            .message_tx
            .send(SessionMessage::Cancel { response_tx })
            .await
            .is_ok()
        {
            // Wait for acknowledgment
            drop(response_rx.await);
        }
    }
}

/// Background worker that processes session messages sequentially
struct SessionWorker {
    session_id: SessionId,
    cwd: PathBuf,
    #[allow(dead_code)]
    client_capabilities: Arc<Mutex<ClientCapabilities>>,
    model: Option<String>,
    mode: Option<String>,
    active_child: Option<tokio::process::Child>,
    cancelled: Arc<AtomicBool>,
    cursor_session_id: Arc<Mutex<Option<String>>>,
}

enum ProcessEventsOutcome {
    Completed,
    Cancelled,
    Interrupted(String),
}

const MAX_INTERRUPTED_RETRIES: usize = 1;
const CHILD_EXIT_GUARD_TIMEOUT_SECS: u64 = 2;
const INTERRUPTED_RETRY_PROMPT: &str =
    "The previous response stream was interrupted. Please continue exactly where you left off, \
without repeating completed steps.";
const INTERRUPTED_RETRY_NOTICE: &str =
    "\n\nConnection interrupted. Attempting automatic resume...\n";

impl SessionWorker {
    fn new(
        session_id: SessionId,
        cwd: PathBuf,
        client_capabilities: Arc<Mutex<ClientCapabilities>>,
        cursor_session_id: Arc<Mutex<Option<String>>>,
    ) -> Self {
        Self {
            session_id,
            cwd,
            client_capabilities,
            model: None,
            mode: None,
            active_child: None,
            cancelled: Arc::new(AtomicBool::new(false)),
            cursor_session_id,
        }
    }

    /// Main loop that processes messages from the channel.
    /// `cancel_rx` is passed in (not stored) so we can pass it to process_events without overlapping borrows.
    async fn run(
        mut self,
        mut message_rx: mpsc::Receiver<SessionMessage>,
        mut cancel_rx: mpsc::Receiver<()>,
    ) {
        info!("Session worker started for {}", self.session_id.0);

        while let Some(msg) = message_rx.recv().await {
            match msg {
                SessionMessage::Prompt {
                    prompt_text,
                    response_tx,
                } => {
                    let result = self.handle_prompt(prompt_text, &mut cancel_rx).await;
                    if let Err(ref e) = result {
                        self.send_agent_text(&format!("\n\nError: {}\n", e.message))
                            .await;
                    }
                    drop(response_tx.send(result));
                }
                SessionMessage::Cancel { response_tx } => {
                    self.handle_cancel().await;
                    let _ = response_tx.send(());
                }
                SessionMessage::SetModel { model } => {
                    self.model = Some(model);
                }
                SessionMessage::SetMode { mode } => {
                    self.mode = Some(mode);
                }
            }
        }

        info!("Session worker stopped for {}", self.session_id.0);
    }

    /// Handle a prompt message
    async fn handle_prompt(
        &mut self,
        prompt_text: String,
        cancel_rx: &mut mpsc::Receiver<()>,
    ) -> Result<StopReason, Error> {
        info!("Received prompt text: {:?}", prompt_text);

        // Drain stale cancel signals from a previous prompt.
        while cancel_rx.try_recv().is_ok() {}
        self.cancelled.store(false, Ordering::SeqCst);

        // Handle /login command
        let trimmed = prompt_text.trim();
        if trimmed == "/login" || trimmed == "login" {
            return self.handle_login_command().await;
        }

        let mut attempts = 0usize;
        let mut attempt_prompt = prompt_text;

        loop {
            info!("Processing prompt: {}", truncate(&attempt_prompt, 100));

            let outcome = self.run_prompt_attempt(&attempt_prompt, cancel_rx).await?;

            match outcome {
                ProcessEventsOutcome::Completed => return Ok(StopReason::EndTurn),
                ProcessEventsOutcome::Cancelled => return Ok(StopReason::Cancelled),
                ProcessEventsOutcome::Interrupted(message) => {
                    // If cancel arrived between attempts, report cancellation instead of retrying.
                    if cancel_rx.try_recv().is_ok() || self.cancelled.load(Ordering::SeqCst) {
                        self.cancelled.store(true, Ordering::SeqCst);
                        return Ok(StopReason::Cancelled);
                    }

                    if attempts < MAX_INTERRUPTED_RETRIES {
                        attempts += 1;
                        warn!(
                            "{}; retrying once (attempt {}/{})",
                            message, attempts, MAX_INTERRUPTED_RETRIES
                        );
                        self.send_agent_text(INTERRUPTED_RETRY_NOTICE).await;
                        attempt_prompt = INTERRUPTED_RETRY_PROMPT.to_string();
                        continue;
                    }

                    error!("{}", message);
                    self.send_error(&message);
                    return Err(Error::internal_error().data(message));
                }
            }
        }
    }

    async fn run_prompt_attempt(
        &mut self,
        prompt_text: &str,
        cancel_rx: &mut mpsc::Receiver<()>,
    ) -> Result<ProcessEventsOutcome, Error> {
        let resume_session_id = self.cursor_session_id.lock().unwrap().clone();

        if let Some(ref sid) = resume_session_id {
            info!("Resuming Cursor conversation (session_id={})", sid);
        } else {
            info!("Starting new Cursor conversation (no previous session_id)");
        }

        let process = CursorProcess::spawn(
            prompt_text,
            Some(self.cwd.as_path()),
            self.model.as_deref(),
            self.mode.as_deref(),
            resume_session_id.as_deref(),
        )
        .await
        .map_err(|e| {
            error!("Failed to spawn Cursor CLI: {e}");
            self.send_error(&format!("Failed to start Cursor CLI: {e}"));
            Error::internal_error().data(e.to_string())
        })?;

        let CursorProcessParts { child, event_rx } = process.into_parts();
        self.active_child = Some(child);

        match self.process_events(event_rx, cancel_rx).await {
            Ok(outcome) => Ok(self.finalize_child_after_attempt(outcome).await),
            Err(err) => {
                // Ensure we never leak an in-flight child process on error paths.
                self.terminate_active_child().await;
                Err(err)
            }
        }
    }

    async fn finalize_child_after_attempt(
        &mut self,
        outcome: ProcessEventsOutcome,
    ) -> ProcessEventsOutcome {
        let Some(mut child) = self.active_child.take() else {
            return outcome;
        };

        match outcome {
            ProcessEventsOutcome::Cancelled => {
                // Ensure cancellation leaves no running child process.
                drop(child.kill().await);
                drop(child.wait().await);
                ProcessEventsOutcome::Cancelled
            }
            ProcessEventsOutcome::Completed => {
                match tokio::time::timeout(
                    Duration::from_secs(CHILD_EXIT_GUARD_TIMEOUT_SECS),
                    child.wait(),
                )
                .await
                {
                    Ok(Ok(status)) => {
                        if !status.success() {
                            warn!("Cursor process exited non-zero after final result: {status}");
                        }
                    }
                    Ok(Err(e)) => {
                        warn!("Failed to wait for Cursor process exit after completion: {e}");
                    }
                    Err(_) => {
                        warn!("Cursor process still running after completion; terminating");
                        drop(child.kill().await);
                        drop(child.wait().await);
                    }
                }
                ProcessEventsOutcome::Completed
            }
            ProcessEventsOutcome::Interrupted(detail) => {
                let detail = match tokio::time::timeout(
                    Duration::from_secs(CHILD_EXIT_GUARD_TIMEOUT_SECS),
                    child.wait(),
                )
                .await
                {
                    Ok(Ok(status)) if status.success() => format!(
                        "{detail}; process exited successfully without terminal result event"
                    ),
                    Ok(Ok(status)) => format!("{detail}; process exited with status {status}"),
                    Ok(Err(e)) => format!("{detail}; failed waiting for process exit: {e}"),
                    Err(_) => {
                        warn!(
                            "Cursor process still running after stream interruption; terminating"
                        );
                        drop(child.kill().await);
                        drop(child.wait().await);
                        format!(
                            "{detail}; process did not exit after stream interruption and was terminated"
                        )
                    }
                };
                ProcessEventsOutcome::Interrupted(detail)
            }
        }
    }

    async fn terminate_active_child(&mut self) {
        if let Some(mut child) = self.active_child.take() {
            drop(child.kill().await);
            drop(child.wait().await);
        }
    }

    /// Handle the /login command
    async fn handle_login_command(&mut self) -> Result<StopReason, Error> {
        info!("Handling /login command");

        match find_cursor_binary() {
            Ok(binary_path) => {
                info!("Found cursor binary at: {:?}", binary_path);
                self.send_agent_text("Launching Cursor login flow...\n")
                    .await;

                let result = tokio::process::Command::new(&binary_path)
                    .arg("login")
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn();

                match result {
                    Ok(mut child) => {
                        info!("Login process spawned successfully");
                        self.send_agent_text(
                            "Browser should open for authentication. Please complete login there.\n",
                        )
                        .await;

                        match child.wait().await {
                            Ok(status) => {
                                info!("Login process exited with status: {}", status);
                                if status.success() {
                                    self.send_agent_text(
                                        "Login completed! You can now use Cursor.\n",
                                    )
                                    .await;
                                } else {
                                    self.send_agent_text(&format!(
                                        "Login process exited with status: {}\n",
                                        status
                                    ))
                                    .await;
                                }
                                Ok(StopReason::EndTurn)
                            }
                            Err(e) => {
                                error!("Failed to wait for login process: {e}");
                                self.send_agent_text(&format!("Error waiting for login: {e}\n"))
                                    .await;
                                Ok(StopReason::EndTurn)
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to spawn login process: {e}");
                        self.send_agent_text(&format!("Failed to launch login: {e}\n"))
                            .await;
                        Err(Error::internal_error().data(e.to_string()))
                    }
                }
            }
            Err(e) => {
                error!("Cursor CLI not found: {e}");
                self.send_agent_text(
                    "Cursor CLI not found. Please install it:\n\
                    curl https://cursor.com/install -fsSL | bash\n",
                )
                .await;
                Err(Error::internal_error().data(e.to_string()))
            }
        }
    }

    /// Kill the active CLI process and mark the session as cancelled.
    async fn handle_cancel(&mut self) {
        info!("Cancelling current operation");
        self.cancelled.store(true, Ordering::SeqCst);
        if let Some(mut child) = self.active_child.take() {
            drop(child.kill().await);
            drop(child.wait().await);
        }
    }

    /// Capture Cursor CLI session ID from events
    fn capture_cursor_session_id_if_present(&self, new_session_id: Option<&String>) {
        if let Some(new_id) = new_session_id {
            let mut current = self.cursor_session_id.lock().unwrap();
            if let Some(ref old_id) = *current {
                if old_id != new_id {
                    warn!(
                        "Cursor session ID changed unexpectedly: {} -> {}",
                        old_id, new_id
                    );
                }
            } else {
                debug!("Captured Cursor session ID: {}", new_id);
            }
            *current = Some(new_id.clone());
        }
    }

    /// Process Cursor CLI events while polling for cancellation.
    async fn process_events(
        &mut self,
        mut event_rx: mpsc::Receiver<StreamEvent>,
        cancel_rx: &mut mpsc::Receiver<()>,
    ) -> Result<ProcessEventsOutcome, Error> {
        loop {
            tokio::select! {
                event = event_rx.recv() => {
                    let Some(event) = event else {
                        break;
                    };
                    match event {
                StreamEvent::System(sys) => {
                    debug!("Cursor system event: subtype={}", sys.subtype);
                    self.capture_cursor_session_id_if_present(sys.session_id.as_ref());
                }
                StreamEvent::Thinking(thinking) => {
                    self.capture_cursor_session_id_if_present(thinking.session_id.as_ref());
                    if let Some(text) = &thinking.text {
                        self.send_agent_thought(text).await;
                    }
                }
                StreamEvent::Assistant(msg) => {
                    for part in &msg.message.content {
                        if let crate::stream_json::ContentPart::Text { text } = part {
                            if text.contains("Authentication required")
                                || text.contains("Please run 'agent login'")
                                || text.contains("Please run `/login`")
                            {
                                info!("Authentication required detected");
                                self.send_agent_text(
                                    "Authentication required. Please use the /login command.\n",
                                )
                                .await;
                                return Err(Error::auth_required());
                            }
                            self.send_agent_text(text).await;
                        }
                    }
                }
                StreamEvent::ToolCall(tc) => {
                    self.capture_cursor_session_id_if_present(tc.session_id.as_ref());
                    let tool_name = tc.get_tool_name().unwrap_or("unknown");
                    let tool_key = tc.get_tool_call_key();
                    let arguments = tc.get_arguments().cloned();
                    let terminal_tool = is_terminal_tool_call(tool_name, tool_key, arguments.as_ref());
                    let terminal_output_supported = self.supports_terminal_output();

                    match tc.subtype {
                        ToolCallSubtype::Started => {
                            if tool_name == "tool" {
                                if let Some(key) = tool_key {
                                    info!("Unknown tool type discovered: {} - please report this so we can add support", key);
                                }
                            }
                            debug!(
                                "Tool call started: {} - {} (key: {:?})",
                                tc.call_id, tool_name, tool_key
                            );

                            let title = format_tool_title(tool_name, tool_key, &arguments);
                            let kind = map_tool_kind(tool_name);

                            let mut tool_call =
                                ToolCall::new(ToolCallId::new(tc.call_id.clone()), title)
                                    .kind(kind)
                                    .status(ToolCallStatus::InProgress);

                            if let Some(args) = &arguments {
                                tool_call = tool_call.raw_input(args.clone());
                            }

                            let locations = extract_tool_locations(&tc);
                            if !locations.is_empty() {
                                tool_call = tool_call.locations(locations);
                            }

                            if terminal_tool && terminal_output_supported {
                                tool_call = tool_call.content(vec![ToolCallContent::Terminal(
                                    Terminal::new(tc.call_id.clone()),
                                )]);
                                tool_call = tool_call.meta(build_terminal_info_meta(
                                    &tc.call_id,
                                    tc.get_working_directory().as_deref(),
                                ));
                            }

                            self.send_tool_call(tool_call).await;
                        }
                        ToolCallSubtype::Completed => {
                            debug!("Tool call completed: {}", tc.call_id);

                            let result_data = tc.get_result_data();
                            let status = if tool_call_failed(&tc, result_data) {
                                ToolCallStatus::Failed
                            } else {
                                ToolCallStatus::Completed
                            };
                            let mut fields = ToolCallUpdateFields::new().status(status);

                            if let Some(result_data) = result_data {
                                fields = fields.raw_output(result_data.clone());
                            } else if let Some(result) = &tc.result {
                                if let Some(output) = &result.output {
                                    let val = serde_json::Value::String(output.clone());
                                    fields = fields.raw_output(val);
                                } else if let Some(error) = &result.error {
                                    let val = serde_json::json!({"error": error});
                                    fields = fields.raw_output(val);
                                }
                            } else if let Some(tool_call) = &tc.tool_call {
                                fields = fields.raw_output(tool_call.clone());
                            }

                            let mut terminal_output: Option<String> = None;
                            let mut terminal_exit_code = 0;
                            if terminal_tool {
                                terminal_output =
                                    extract_terminal_output(&tc, result_data).as_deref().map(str::to_owned);
                                terminal_exit_code = infer_terminal_exit_code(&tc, result_data);

                                if terminal_output.is_none() {
                                    terminal_output =
                                        result_data.and_then(pretty_json_for_terminal_fallback);
                                }
                                if terminal_output.is_none() {
                                    terminal_output = tc
                                        .tool_call
                                        .as_ref()
                                        .and_then(pretty_json_for_terminal_fallback);
                                }

                                if !terminal_output_supported {
                                    if let Some(output) = terminal_output.as_ref() {
                                        if !output.is_empty() {
                                            fields = fields.content(vec![ToolCallContent::from(
                                                wrap_terminal_output_for_markdown(output),
                                            )]);
                                        }
                                    }
                                }
                            }

                            let update =
                                ToolCallUpdate::new(ToolCallId::new(tc.call_id.clone()), fields);

                            if terminal_tool && terminal_output_supported {
                                if let Some(output) = terminal_output.as_deref() {
                                    if !output.is_empty() {
                                        let output_update = ToolCallUpdate::new(
                                            ToolCallId::new(tc.call_id.clone()),
                                            ToolCallUpdateFields::new(),
                                        )
                                        .meta(build_terminal_output_meta(&tc.call_id, output));
                                        self.send_tool_call_update(output_update).await;
                                    }
                                }

                                let update =
                                    update.meta(build_terminal_exit_meta(&tc.call_id, terminal_exit_code));
                                self.send_tool_call_update(update).await;
                            } else {
                                self.send_tool_call_update(update).await;
                            }
                        }
                    }
                }
                StreamEvent::Result(res) => {
                    self.capture_cursor_session_id_if_present(res.session_id.as_ref());

                    if res.is_error {
                        if let Some(error) = &res.error {
                            if error.contains("Authentication required")
                                || error.contains("Please run 'agent login'")
                            {
                                info!("Authentication required detected in error");
                                self.send_agent_text(
                                    "Authentication required. Please use the /login command.\n",
                                )
                                .await;
                                return Err(Error::auth_required());
                            }

                            error!("Cursor error: {}", error);
                            self.send_agent_text(&format!("\n\nError: {}", error)).await;
                        }
                        return Ok(ProcessEventsOutcome::Completed);
                    }
                    debug!(
                        "Cursor completed: duration={}ms",
                        res.duration_ms.unwrap_or(0)
                    );
                    return Ok(ProcessEventsOutcome::Completed);
                }
                StreamEvent::User(_) => {
                    // User messages are echoed back, ignore
                }
            }
                }
                cancel_msg = cancel_rx.recv() => {
                    if cancel_msg.is_some() {
                        info!("Cancel requested, killing CLI process");
                        self.cancelled.store(true, Ordering::SeqCst);
                        if let Some(mut child) = self.active_child.take() {
                            drop(child.kill().await);
                            drop(child.wait().await);
                        }
                        return Ok(ProcessEventsOutcome::Cancelled);
                    }
                }
            }
        }

        if self.cancelled.load(Ordering::SeqCst) {
            Ok(ProcessEventsOutcome::Cancelled)
        } else {
            warn!("Cursor event stream closed before terminal result event");
            Ok(ProcessEventsOutcome::Interrupted(
                "Cursor event stream closed before terminal result event".to_string(),
            ))
        }
    }

    fn send_error(&self, message: &str) {
        let session_id = self.session_id.clone();
        let message = message.to_string();
        tokio::task::spawn_local(async move {
            if let Some(client) = ACP_CLIENT.get() {
                drop(
                    client
                        .session_notification(SessionNotification::new(
                            session_id,
                            SessionUpdate::AgentMessageChunk(ContentChunk::new(message.into())),
                        ))
                        .await,
                );
            }
        });
    }

    async fn send_agent_text(&self, text: &str) {
        if let Some(client) = ACP_CLIENT.get() {
            if let Err(e) = client
                .session_notification(SessionNotification::new(
                    self.session_id.clone(),
                    SessionUpdate::AgentMessageChunk(ContentChunk::new(text.to_string().into())),
                ))
                .await
            {
                error!("Failed to send agent text: {:?}", e);
            }
        }
    }

    async fn send_agent_thought(&self, text: &str) {
        if let Some(client) = ACP_CLIENT.get() {
            if let Err(e) = client
                .session_notification(SessionNotification::new(
                    self.session_id.clone(),
                    SessionUpdate::AgentThoughtChunk(ContentChunk::new(text.to_string().into())),
                ))
                .await
            {
                error!("Failed to send agent thought: {:?}", e);
            }
        }
    }

    async fn send_tool_call(&self, tool_call: ToolCall) {
        if let Some(client) = ACP_CLIENT.get() {
            if let Err(e) = client
                .session_notification(SessionNotification::new(
                    self.session_id.clone(),
                    SessionUpdate::ToolCall(tool_call),
                ))
                .await
            {
                error!("Failed to send tool call: {:?}", e);
            }
        }
    }

    async fn send_tool_call_update(&self, update: ToolCallUpdate) {
        if let Some(client) = ACP_CLIENT.get() {
            if let Err(e) = client
                .session_notification(SessionNotification::new(
                    self.session_id.clone(),
                    SessionUpdate::ToolCallUpdate(update),
                ))
                .await
            {
                error!("Failed to send tool call update: {:?}", e);
            }
        }
    }

    fn supports_terminal_output(&self) -> bool {
        let caps = self.client_capabilities.lock().unwrap();
        if let Some(enabled) = caps
            .meta
            .as_ref()
            .and_then(|meta| meta.get("terminal_output"))
            .and_then(|value| value.as_bool())
        {
            return enabled;
        }

        // Fallback when terminal_output meta is not advertised.
        caps.terminal
    }
}

use crate::stream_json::ToolCallEvent;

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

/// Extract file locations from a tool call for the "follow the agent" feature.
/// This enables clients to track which files the agent is working with in real-time.
fn extract_tool_locations(tc: &ToolCallEvent) -> Vec<ToolCallLocation> {
    let mut locations = Vec::new();

    // Extract primary file path
    if let Some(path) = tc.get_file_path() {
        let mut location = ToolCallLocation::new(path);

        // Add line number if available
        if let Some(line) = tc.get_line_number() {
            location = location.line(line);
        }

        locations.push(location);
    }

    // For shell commands, add working directory as a location
    if let Some(tool_key) = tc.get_tool_call_key() {
        if tool_key == "shellToolCall" || tool_key == "bashToolCall" {
            if let Some(cwd) = tc.get_working_directory() {
                // Only add if not already present
                if !locations.iter().any(|l| l.path.to_string_lossy() == cwd) {
                    locations.push(ToolCallLocation::new(cwd));
                }
            }
        }
    }

    locations
}

fn is_terminal_tool_call(
    tool_name: &str,
    tool_key: Option<&str>,
    arguments: Option<&serde_json::Value>,
) -> bool {
    let normalized_name = tool_name.to_ascii_lowercase();
    if matches!(
        normalized_name.as_str(),
        "shell" | "bash" | "execute" | "run" | "terminal"
    ) || normalized_name.contains("shell")
        || normalized_name.contains("bash")
        || normalized_name.contains("terminal")
        || normalized_name.contains("execute")
    {
        return true;
    }

    if let Some(tool_key) = tool_key {
        let normalized_key = tool_key.to_ascii_lowercase();
        return matches!(
            normalized_key.as_str(),
            "shelltoolcall" | "bashtoolcall" | "terminaltoolcall" | "executetoolcall"
        ) || normalized_key.contains("shell")
            || normalized_key.contains("bash")
            || normalized_key.contains("terminal")
            || normalized_key.contains("execute")
            || normalized_key.contains("command")
            || normalized_key.contains("run");
    }

    if let Some(args) = arguments.and_then(|v| v.as_object()) {
        for key in [
            "command",
            "cmd",
            "workingDirectory",
            "cwd",
            "timeout",
            "shell",
            "stdin",
            "terminalId",
            "outputByteLimit",
        ] {
            if args.contains_key(key) {
                return true;
            }
        }
    }

    false
}

fn tool_call_failed(tc: &ToolCallEvent, result_data: Option<&serde_json::Value>) -> bool {
    if let Some(result) = &tc.result {
        if !result.success {
            return true;
        }
        if result.error.as_deref().is_some_and(|e| !e.is_empty()) {
            return true;
        }
    }

    if let Some(result_data) = result_data {
        if let Some(success) = result_data.get("success").and_then(|v| v.as_bool()) {
            if !success {
                return true;
            }
        }
    }

    false
}

fn extract_terminal_output(
    tc: &ToolCallEvent,
    result_data: Option<&serde_json::Value>,
) -> Option<String> {
    if let Some(result) = &tc.result {
        if let Some(output) = &result.output {
            if !output.is_empty() {
                return Some(output.clone());
            }
        }
        if let Some(error) = &result.error {
            if !error.is_empty() {
                return Some(error.clone());
            }
        }
    }

    if let Some(output) = result_data.and_then(extract_terminal_output_from_value) {
        return Some(output);
    }

    if let Some(tool_call) = &tc.tool_call {
        if let Some(output) = extract_terminal_output_from_value(tool_call) {
            return Some(output);
        }
    }

    None
}

fn infer_terminal_exit_code(tc: &ToolCallEvent, result_data: Option<&serde_json::Value>) -> u32 {
    if let Some(data) = result_data {
        if let Some(code) = extract_exit_code_from_value(data) {
            return code;
        }
    }

    if let Some(tool_call) = &tc.tool_call {
        if let Some(code) = extract_exit_code_from_value(tool_call) {
            return code;
        }
    }

    if let Some(result) = &tc.result {
        return if result.success { 0 } else { 1 };
    }

    if tool_call_failed(tc, result_data) {
        1
    } else {
        0
    }
}

fn extract_terminal_output_from_value(value: &serde_json::Value) -> Option<String> {
    let mut chunks = Vec::new();
    collect_terminal_output_chunks(value, &mut chunks);
    let mut normalized = Vec::new();
    for chunk in chunks {
        if chunk.is_empty() {
            continue;
        }
        if normalized
            .last()
            .map(|last: &String| last == &chunk)
            .unwrap_or(false)
        {
            continue;
        }
        normalized.push(chunk);
    }

    if normalized.is_empty() {
        None
    } else {
        Some(normalized.join("\n"))
    }
}

fn collect_terminal_output_chunks(value: &serde_json::Value, chunks: &mut Vec<String>) {
    match value {
        serde_json::Value::String(s) => {
            if !s.is_empty() {
                chunks.push(s.clone());
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_terminal_output_chunks(value, chunks);
            }
        }
        serde_json::Value::Object(obj) => {
            let mut matched_known_key = false;
            for key in [
                "stdout", "stderr", "output", "result", "error", "message", "data", "text",
                "content", "lines", "line", "chunks", "messages", "value", "response", "body",
            ] {
                if let Some(value) = obj.get(key) {
                    matched_known_key = true;
                    collect_terminal_output_chunks(value, chunks);
                }
            }

            // If no known key matched, recurse all values (minus obvious metadata keys).
            if !matched_known_key {
                for (key, value) in obj {
                    if should_skip_terminal_fallback_key(key) {
                        continue;
                    }
                    collect_terminal_output_chunks(value, chunks);
                }
            }
        }
        _ => {}
    }
}

fn should_skip_terminal_fallback_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    matches!(
        key.as_str(),
        "command"
            | "workingdirectory"
            | "cwd"
            | "path"
            | "filepath"
            | "line"
            | "linenumber"
            | "startline"
            | "toolcallid"
            | "call_id"
            | "kind"
            | "status"
            | "title"
            | "name"
            | "description"
            | "args"
            | "arguments"
    )
}

fn extract_exit_code_from_value(value: &serde_json::Value) -> Option<u32> {
    match value {
        serde_json::Value::Object(obj) => {
            for key in ["exit_code", "exitCode", "code"] {
                if let Some(raw) = obj.get(key) {
                    if let Some(code) = raw.as_u64() {
                        return Some(code as u32);
                    }
                    if let Some(code) = raw.as_i64() {
                        return Some(code.max(0) as u32);
                    }
                }
            }
            for key in [
                "result",
                "output",
                "data",
                "error",
                "exit_status",
                "exitStatus",
                "response",
                "body",
            ] {
                if let Some(nested) = obj.get(key) {
                    if let Some(code) = extract_exit_code_from_value(nested) {
                        return Some(code);
                    }
                }
            }
            None
        }
        serde_json::Value::Array(values) => values.iter().find_map(extract_exit_code_from_value),
        _ => None,
    }
}

fn pretty_json_for_terminal_fallback(value: &serde_json::Value) -> Option<String> {
    if value.is_null() {
        return None;
    }

    match value {
        serde_json::Value::String(s) => {
            let trimmed = s.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        _ => serde_json::to_string_pretty(value)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s != "null"),
    }
}

fn wrap_terminal_output_for_markdown(output: &str) -> String {
    let body = output.trim_end_matches('\n');

    // Use a fence longer than any run of backticks in output.
    let mut longest_backtick_run = 0usize;
    let mut current_backtick_run = 0usize;
    for ch in body.chars() {
        if ch == '`' {
            current_backtick_run += 1;
            longest_backtick_run = longest_backtick_run.max(current_backtick_run);
        } else {
            current_backtick_run = 0;
        }
    }

    let fence = "`".repeat((longest_backtick_run + 1).max(3));
    format!("{fence}text\n{body}\n{fence}")
}

fn build_terminal_info_meta(
    terminal_id: &str,
    cwd: Option<&str>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut meta = serde_json::Map::new();

    let mut terminal_info = serde_json::Map::new();
    terminal_info.insert(
        "terminal_id".to_string(),
        serde_json::Value::String(terminal_id.to_string()),
    );
    if let Some(cwd) = cwd {
        terminal_info.insert(
            "cwd".to_string(),
            serde_json::Value::String(cwd.to_string()),
        );
    }
    meta.insert(
        "terminal_info".to_string(),
        serde_json::Value::Object(terminal_info),
    );

    meta
}

fn build_terminal_output_meta(
    terminal_id: &str,
    output: &str,
) -> serde_json::Map<String, serde_json::Value> {
    let mut meta = serde_json::Map::new();

    let mut terminal_output = serde_json::Map::new();
    terminal_output.insert(
        "terminal_id".to_string(),
        serde_json::Value::String(terminal_id.to_string()),
    );
    terminal_output.insert(
        "data".to_string(),
        serde_json::Value::String(output.to_string()),
    );
    meta.insert(
        "terminal_output".to_string(),
        serde_json::Value::Object(terminal_output),
    );

    meta
}

fn build_terminal_exit_meta(
    terminal_id: &str,
    exit_code: u32,
) -> serde_json::Map<String, serde_json::Value> {
    let mut meta = serde_json::Map::new();

    let mut terminal_exit = serde_json::Map::new();
    terminal_exit.insert(
        "terminal_id".to_string(),
        serde_json::Value::String(terminal_id.to_string()),
    );
    terminal_exit.insert(
        "exit_code".to_string(),
        serde_json::Value::Number(serde_json::Number::from(exit_code)),
    );
    terminal_exit.insert("signal".to_string(), serde_json::Value::Null);
    meta.insert(
        "terminal_exit".to_string(),
        serde_json::Value::Object(terminal_exit),
    );

    meta
}

fn format_tool_title(
    tool_name: &str,
    tool_key: Option<&str>,
    arguments: &Option<serde_json::Value>,
) -> String {
    match tool_name {
        "readFile" | "read_file" => {
            if let Some(args) = arguments {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    return format!("Read {}", path);
                }
            }
            "Read file".to_string()
        }
        "writeFile" | "write_file" => {
            if let Some(args) = arguments {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    return format!("Write {}", path);
                }
            }
            "Write file".to_string()
        }
        "edit" => {
            if let Some(args) = arguments {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    return format!("Edit {}", path);
                }
            }
            "Edit file".to_string()
        }
        "delete" => {
            if let Some(args) = arguments {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    return format!("Delete {}", path);
                }
            }
            "Delete file".to_string()
        }
        "shell" | "bash" | "execute" => {
            if let Some(args) = arguments {
                if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
                    return format!("Run: {}", truncate(cmd, 50));
                }
            }
            "Run command".to_string()
        }
        "ls" => {
            if let Some(args) = arguments {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    return format!("List {}", path);
                }
                if let Some(dir) = args.get("directory").and_then(|v| v.as_str()) {
                    return format!("List {}", dir);
                }
            }
            "List directory".to_string()
        }
        "grep" => {
            if let Some(args) = arguments {
                if let Some(pattern) = args.get("pattern").and_then(|v| v.as_str()) {
                    return format!("Grep: {}", truncate(pattern, 30));
                }
            }
            "Search with grep".to_string()
        }
        "glob" => {
            if let Some(args) = arguments {
                if let Some(pattern) = args.get("pattern").and_then(|v| v.as_str()) {
                    return format!("Glob: {}", truncate(pattern, 30));
                }
            }
            "Find files".to_string()
        }
        "search" | "find" => {
            if let Some(args) = arguments {
                if let Some(query) = args.get("query").and_then(|v| v.as_str()) {
                    return format!("Search: {}", truncate(query, 30));
                }
            }
            "Search".to_string()
        }
        "fetch" => {
            if let Some(args) = arguments {
                if let Some(url) = args.get("url").and_then(|v| v.as_str()) {
                    return format!("Fetch {}", truncate(url, 40));
                }
            }
            "Fetch URL".to_string()
        }
        "think" => "Thinking...".to_string(),
        // For unknown tools, try to make a readable title from the raw key
        "tool" => {
            if let Some(key) = tool_key {
                // Convert "somethingToolCall" to "Something"
                let name = key
                    .strip_suffix("ToolCall")
                    .or_else(|| key.strip_suffix("Tool"))
                    .unwrap_or(key);
                // Capitalize first letter
                let mut chars = name.chars();
                match chars.next() {
                    Some(first) => {
                        format!("{}{}", first.to_uppercase(), chars.as_str())
                    }
                    None => "Tool".to_string(),
                }
            } else {
                "Tool".to_string()
            }
        }
        _ => tool_name.to_string(),
    }
}

fn map_tool_kind(tool_name: &str) -> ToolKind {
    match tool_name {
        // Read operations
        "readFile" | "read_file" | "read" => ToolKind::Read,
        // Edit/write operations
        "writeFile" | "write_file" | "write" | "edit" | "patch" => ToolKind::Edit,
        // Delete operations
        "delete" | "remove" | "rm" => ToolKind::Delete,
        // Execute operations
        "shell" | "bash" | "execute" | "run" | "terminal" => ToolKind::Execute,
        // Search operations
        "search" | "grep" | "glob" | "find" | "ls" => ToolKind::Search,
        // Fetch operations
        "fetch" | "http" | "web" | "curl" => ToolKind::Fetch,
        // Think operations
        "think" | "reasoning" => ToolKind::Think,
        // Default
        _ => ToolKind::Other,
    }
}
