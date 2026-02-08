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

use crate::cursor_process::find_cursor_binary;
use agent_client_protocol::{
    Client, ClientCapabilities, ContentBlock, ContentChunk, Error, ImageContent, PromptRequest,
    SessionId, SessionNotification, SessionUpdate, StopReason, TextContent, ToolCall, ToolCallId,
    ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
    WriteTextFileRequest,
};
use tokio::sync::oneshot;
use tracing::{debug, error, info, warn};

use crate::cursor_process::{CursorProcess, CursorProcessParts};
use crate::stream_json::{StreamEvent, ToolCallEvent, ToolCallSubtype};
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
    client_capabilities: Arc<Mutex<ClientCapabilities>>,
    model: Option<String>,
    mode: Option<String>,
    active_child: Option<tokio::process::Child>,
    cancelled: Arc<AtomicBool>,
    cursor_session_id: Arc<Mutex<Option<String>>>,
}

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

        info!("Processing prompt: {}", truncate(&prompt_text, 100));

        let resume_session_id = self.cursor_session_id.lock().unwrap().clone();

        if let Some(ref sid) = resume_session_id {
            info!("Resuming Cursor conversation (session_id={})", sid);
        } else {
            info!("Starting new Cursor conversation (no previous session_id)");
        }

        let process = CursorProcess::spawn(
            &prompt_text,
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

        let result = self.process_events(event_rx, cancel_rx).await;

        if let Some(mut child) = self.active_child.take() {
            drop(child.kill().await);
            drop(child.wait().await);
        }

        // Pick up any cancel signal that arrived after process_events finished.
        if cancel_rx.try_recv().is_ok() {
            self.cancelled.store(true, Ordering::SeqCst);
        }

        if self.cancelled.load(Ordering::SeqCst) {
            return Ok(StopReason::Cancelled);
        }

        result
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

    /// Process events from Cursor CLI.
    /// Uses select! so we can react to cancel requests even while waiting for the next event.
    async fn process_events(
        &mut self,
        mut event_rx: mpsc::Receiver<StreamEvent>,
        cancel_rx: &mut mpsc::Receiver<()>,
    ) -> Result<StopReason, Error> {
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

                            self.send_tool_call(tool_call).await;
                        }
                        ToolCallSubtype::Completed => {
                            debug!("Tool call completed: {}", tc.call_id);

                            let mut fields =
                                ToolCallUpdateFields::new().status(ToolCallStatus::Completed);

                            if let Some(result_data) = tc.get_result_data() {
                                fields = fields.raw_output(result_data.clone());
                            } else if let Some(result) = &tc.result {
                                if let Some(output) = &result.output {
                                    let val = serde_json::Value::String(output.clone());
                                    fields = fields.raw_output(val);
                                } else if let Some(error) = &result.error {
                                    let val = serde_json::json!({"error": error});
                                    fields = fields.raw_output(val);
                                }
                            }

                            self.send_tool_call_update(ToolCallUpdate::new(
                                ToolCallId::new(tc.call_id.clone()),
                                fields,
                            ))
                            .await;

                            self.notify_file_written(&tc).await;
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
                        return Ok(StopReason::EndTurn);
                    }
                    debug!(
                        "Cursor completed: duration={}ms",
                        res.duration_ms.unwrap_or(0)
                    );
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
                        return Ok(StopReason::Cancelled);
                    }
                }
            }
        }

        if self.cancelled.load(Ordering::SeqCst) {
            Ok(StopReason::Cancelled)
        } else {
            Ok(StopReason::EndTurn)
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

    // Notify the client about a file change so it can track it in its action log.
    async fn notify_file_written(&self, tc: &ToolCallEvent) {
        if !self.client_supports_write() {
            return;
        }

        let Some(tool_key) = tc.get_tool_call_key() else {
            return;
        };

        let path = match tool_key {
            "writeToolCall" | "editToolCall" | "deleteToolCall" => {
                tc.get_file_path().map(|p| self.resolve_path(&p))
            }
            _ => None,
        };

        let Some(path) = path else {
            return;
        };

        // For deletes, send empty content so the client knows the file was removed.
        // For writes/edits, read the current content from disk (the CLI already wrote it).
        let content = if tool_key == "deleteToolCall" {
            String::new()
        } else {
            match Self::read_file_for_notification(&path).await {
                Some(c) => c,
                None => return,
            }
        };

        let Some(client) = ACP_CLIENT.get() else {
            return;
        };

        let request = WriteTextFileRequest::new(self.session_id.clone(), &path, content);
        if let Err(e) = client.write_text_file(request).await {
            debug!("client.write_text_file failed for {}: {e}", path.display());
        }
    }

    //// Read a file for notification, skipping files that are too large or non-text.
    const MAX_NOTIFY_FILE_SIZE: u64 = 10 * 1024 * 1024; // 10 MiB

    async fn read_file_for_notification(path: &PathBuf) -> Option<String> {
        let metadata = match tokio::fs::metadata(path).await {
            Ok(m) => m,
            Err(e) => {
                debug!("Could not stat file {}: {e}", path.display());
                return None;
            }
        };

        if metadata.len() > Self::MAX_NOTIFY_FILE_SIZE {
            debug!(
                "Skipping notification for {} ({} bytes exceeds limit)",
                path.display(),
                metadata.len()
            );
            return None;
        }

        match tokio::fs::read_to_string(path).await {
            Ok(c) => Some(c),
            Err(e) => {
                debug!("Could not read file {}: {e}", path.display());
                None
            }
        }
    }

    fn client_supports_write(&self) -> bool {
        self.client_capabilities.lock().unwrap().fs.write_text_file
    }

    fn resolve_path(&self, p: &str) -> PathBuf {
        let path = PathBuf::from(p);
        if path.is_absolute() {
            path
        } else {
            self.cwd.join(path)
        }
    }
}

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
