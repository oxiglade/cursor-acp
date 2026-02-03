//! Session management for Cursor ACP.
//!
//! This module handles individual sessions, spawning Cursor CLI processes,
//! and translating events to ACP session updates.

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::cursor_process::find_cursor_binary;
use agent_client_protocol::{
    Client, ClientCapabilities, ContentBlock, ContentChunk, Error, PromptRequest, SessionId,
    SessionNotification, SessionUpdate, StopReason, TextContent, ToolCall, ToolCallId,
    ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, error, info};

use crate::cursor_process::CursorProcess;
use crate::stream_json::{StreamEvent, ToolCallSubtype};
use crate::ACP_CLIENT;

/// A session represents an active conversation with Cursor
pub struct Session {
    session_id: SessionId,
    cwd: PathBuf,
    #[allow(dead_code)]
    client_capabilities: Arc<Mutex<ClientCapabilities>>,
    model: RefCell<Option<String>>,
    mode: RefCell<Option<String>>,
    active_process: TokioMutex<Option<CursorProcess>>,
    /// Cursor CLI's internal session ID for conversation continuity
    cursor_session_id: RefCell<Option<String>>,
}

impl Session {
    pub fn new(
        session_id: SessionId,
        cwd: PathBuf,
        client_capabilities: Arc<Mutex<ClientCapabilities>>,
    ) -> Self {
        Self {
            session_id,
            cwd,
            client_capabilities,
            model: RefCell::new(None),
            mode: RefCell::new(None),
            active_process: TokioMutex::new(None),
            cursor_session_id: RefCell::new(None),
        }
    }

    pub fn set_model(&self, model: String) {
        *self.model.borrow_mut() = Some(model);
    }

    pub fn set_mode(&self, mode: String) {
        *self.mode.borrow_mut() = Some(mode);
    }

    pub fn set_cursor_session_id(&self, id: String) {
        *self.cursor_session_id.borrow_mut() = Some(id);
    }

    pub fn cursor_session_id(&self) -> Option<String> {
        self.cursor_session_id.borrow().clone()
    }

    /// Process a prompt request by spawning Cursor CLI
    pub async fn prompt(&self, request: PromptRequest) -> Result<StopReason, Error> {
        let prompt_text = request
            .prompt
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(TextContent { text, .. }) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if prompt_text.is_empty() {
            return Ok(StopReason::EndTurn);
        }

        info!("Received prompt text: {:?}", prompt_text);

        // Handle /login command (with or without leading slash)
        let trimmed = prompt_text.trim();
        if trimmed == "/login" || trimmed == "login" {
            return self.handle_login_command().await;
        }

        info!("Processing prompt: {}", truncate(&prompt_text, 100));

        let model = self.model.borrow().clone();
        let mode = self.mode.borrow().clone();
        let resume_session_id = self.cursor_session_id.borrow().clone();

        let mut process = CursorProcess::spawn(
            &prompt_text,
            Some(self.cwd.as_path()),
            model.as_deref(),
            mode.as_deref(),
            resume_session_id.as_deref(),
        )
        .await
        .map_err(|e| {
            error!("Failed to spawn Cursor CLI: {e}");
            self.send_error(&format!("Failed to start Cursor CLI: {e}"));
            Error::internal_error().data(e.to_string())
        })?;

        let result = self.process_events(&mut process).await;
        drop(process.wait().await);

        // Clear active process
        *self.active_process.lock().await = None;

        result
    }

    /// Handle the /login command by launching the cursor-agent login flow
    async fn handle_login_command(&self) -> Result<StopReason, Error> {
        info!("Handling /login command");

        match find_cursor_binary() {
            Ok(binary_path) => {
                info!("Found cursor binary at: {:?}", binary_path);
                self.send_agent_text("Launching Cursor login flow...\n")
                    .await;

                // Launch the login command with output capture for debugging
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

    /// Cancel the current operation
    pub async fn cancel(&self) {
        if let Some(mut process) = self.active_process.lock().await.take() {
            drop(process.kill().await);
        }
    }

    /// Process events from Cursor CLI and send ACP updates
    async fn process_events(&self, process: &mut CursorProcess) -> Result<StopReason, Error> {
        while let Some(event) = process.next_event().await {
            match event {
                StreamEvent::System(sys) => {
                    debug!("Cursor system event: subtype={}", sys.subtype);
                }
                StreamEvent::Thinking(thinking) => {
                    // Send thinking chunks as agent thought
                    if let Some(text) = &thinking.text {
                        self.send_agent_thought(text).await;
                    }
                }
                StreamEvent::Assistant(msg) => {
                    for part in &msg.message.content {
                        if let crate::stream_json::ContentPart::Text { text } = part {
                            // Check for authentication required message
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
                StreamEvent::ToolCall(tc) => match tc.subtype {
                    ToolCallSubtype::Started => {
                        let tool_name = tc.tool_name.as_deref().unwrap_or("unknown");
                        debug!("Tool call started: {} - {}", tc.call_id, tool_name);

                        let mut tool_call = ToolCall::new(
                            ToolCallId::new(tc.call_id.clone()),
                            format_tool_title(tool_name, &tc.arguments),
                        )
                        .kind(map_tool_kind(tool_name))
                        .status(ToolCallStatus::InProgress);

                        if let Some(args) = &tc.arguments {
                            tool_call = tool_call.raw_input(args.clone());
                        }
                        self.send_tool_call(tool_call).await;
                    }
                    ToolCallSubtype::Completed => {
                        debug!("Tool call completed: {}", tc.call_id);

                        let mut fields =
                            ToolCallUpdateFields::new().status(ToolCallStatus::Completed);

                        if let Some(result) = &tc.result {
                            if let Some(output) = &result.output {
                                fields =
                                    fields.raw_output(serde_json::Value::String(output.clone()));
                            }
                            if let Some(error) = &result.error {
                                fields = fields.raw_output(serde_json::json!({"error": error}));
                            }
                        }

                        self.send_tool_call_update(ToolCallUpdate::new(
                            ToolCallId::new(tc.call_id.clone()),
                            fields,
                        ))
                        .await;
                    }
                },
                StreamEvent::Result(res) => {
                    // Capture the Cursor CLI session ID for conversation continuity
                    if let Some(cursor_sid) = &res.session_id {
                        *self.cursor_session_id.borrow_mut() = Some(cursor_sid.clone());
                        debug!("Captured Cursor session ID: {}", cursor_sid);
                    }

                    if res.is_error {
                        if let Some(error) = &res.error {
                            // Check for authentication error
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
                        // Note: StopReason doesn't have Error variant, so we return EndTurn
                        // The error message is sent via agent text above
                        return Ok(StopReason::EndTurn);
                    }
                    debug!(
                        "Cursor completed: duration={}ms",
                        res.duration_ms.unwrap_or(0)
                    );
                }
                StreamEvent::User(_) => {
                    // User messages are echoed back, we can ignore
                }
            }
        }

        Ok(StopReason::EndTurn)
    }

    fn send_error(&self, message: &str) {
        // Fire and forget error notification
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
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

fn format_tool_title(tool_name: &str, arguments: &Option<serde_json::Value>) -> String {
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
        "runCommand" | "run_command" | "execute" => {
            if let Some(args) = arguments {
                if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
                    return format!("Run: {}", truncate(cmd, 50));
                }
            }
            "Run command".to_string()
        }
        _ => tool_name.to_string(),
    }
}

fn map_tool_kind(tool_name: &str) -> ToolKind {
    match tool_name {
        "readFile" | "read_file" => ToolKind::Read,
        "writeFile" | "write_file" | "edit" | "patch" => ToolKind::Edit,
        "runCommand" | "run_command" | "execute" | "shell" => ToolKind::Execute,
        "search" | "grep" | "find" => ToolKind::Search,
        "fetch" | "http" | "web" => ToolKind::Fetch,
        _ => ToolKind::Other,
    }
}
