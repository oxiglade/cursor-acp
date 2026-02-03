//! Cursor agent implementation for ACP.
//!
//! This module implements the ACP Agent trait, bridging between
//! ACP protocol messages and the Cursor CLI.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use agent_client_protocol::{
    Agent, AgentCapabilities, AuthMethod, AuthMethodId, AuthenticateRequest, AuthenticateResponse,
    AvailableCommand, AvailableCommandsUpdate, CancelNotification, Client, ClientCapabilities,
    Error, Implementation, InitializeRequest, InitializeResponse, ListSessionsRequest,
    ListSessionsResponse, LoadSessionRequest, LoadSessionResponse, ModelInfo, NewSessionRequest,
    NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse, ProtocolVersion,
    SessionCapabilities, SessionId, SessionInfo, SessionListCapabilities, SessionMode,
    SessionModeState, SessionModelState, SessionNotification, SessionUpdate,
    SetSessionConfigOptionRequest, SetSessionConfigOptionResponse, SetSessionModeRequest,
    SetSessionModeResponse, SetSessionModelRequest, SetSessionModelResponse,
};
use tracing::{debug, info};

use crate::cursor_process::find_cursor_binary;
use crate::session::Session;
use crate::session_storage::{generate_title, SessionMetadata, SessionStorage};
use crate::ACP_CLIENT;

/// Authentication method for Cursor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorAuthMethod {
    /// Browser login via `cursor-agent login`
    BrowserLogin,
    /// Use CURSOR_API_KEY environment variable
    ApiKey,
}

impl From<CursorAuthMethod> for AuthMethodId {
    fn from(method: CursorAuthMethod) -> Self {
        Self::new(match method {
            CursorAuthMethod::BrowserLogin => "cursor-login",
            CursorAuthMethod::ApiKey => "cursor-api-key",
        })
    }
}

impl TryFrom<AuthMethodId> for CursorAuthMethod {
    type Error = Error;

    fn try_from(value: AuthMethodId) -> Result<Self, Self::Error> {
        match value.0.as_ref() {
            "cursor-login" => Ok(CursorAuthMethod::BrowserLogin),
            "cursor-api-key" => Ok(CursorAuthMethod::ApiKey),
            _ => Err(Error::invalid_params().data("unsupported authentication method")),
        }
    }
}

/// The Cursor ACP agent
pub struct CursorAgent {
    /// Capabilities of the connected client
    client_capabilities: Arc<Mutex<ClientCapabilities>>,
    /// Active sessions mapped by `SessionId`
    sessions: Rc<RefCell<HashMap<SessionId, Rc<Session>>>>,
    /// Session working directories
    session_roots: Arc<Mutex<HashMap<SessionId, PathBuf>>>,
    /// Counter for generating session IDs
    next_session_id: RefCell<u64>,
    /// Whether authentication has been completed
    authenticated: RefCell<bool>,
    /// Persistent session storage for history
    session_storage: RefCell<SessionStorage>,
}

impl CursorAgent {
    /// Create a new Cursor agent
    pub fn new() -> Self {
        // Check if already authenticated (has API key or cached credentials)
        let authenticated = std::env::var("CURSOR_API_KEY").is_ok();

        // Load persistent session storage
        let session_storage = SessionStorage::load();
        info!(
            "Loaded {} sessions from storage",
            session_storage.list().len()
        );

        // Initialize next_session_id to avoid reusing existing session IDs
        let max_existing_id = session_storage
            .list()
            .iter()
            .filter_map(|meta| {
                meta.session_id
                    .strip_prefix("cursor-session-")
                    .and_then(|s| s.parse::<u64>().ok())
            })
            .max()
            .unwrap_or(0);

        debug!(
            "Next session ID will be {} (max existing: {})",
            max_existing_id + 1,
            max_existing_id
        );

        Self {
            client_capabilities: Arc::default(),
            sessions: Rc::default(),
            session_roots: Arc::default(),
            next_session_id: RefCell::new(max_existing_id + 1),
            authenticated: RefCell::new(authenticated),
            session_storage: RefCell::new(session_storage),
        }
    }

    fn get_session(&self, session_id: &SessionId) -> Result<Rc<Session>, Error> {
        self.sessions
            .borrow()
            .get(session_id)
            .cloned()
            .ok_or_else(|| Error::resource_not_found(None))
    }

    fn next_session_id(&self) -> SessionId {
        let id = *self.next_session_id.borrow();
        *self.next_session_id.borrow_mut() = id + 1;
        SessionId::new(format!("cursor-session-{id}"))
    }

    /// Build auth methods based on client capabilities
    fn build_auth_methods(&self) -> Vec<AuthMethod> {
        let caps = self.client_capabilities.lock().unwrap();

        // Check if client supports terminal-auth capability
        let supports_terminal_auth = caps
            .meta
            .as_ref()
            .and_then(|m| m.get("terminal-auth"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mut methods = Vec::new();

        let mut login_method = AuthMethod::new(CursorAuthMethod::BrowserLogin, "Login with Cursor")
            .description("Opens a browser to authenticate with your Cursor account");

        // If client supports terminal-auth, add metadata for launching the login command
        if supports_terminal_auth {
            if let Ok(cli_path) = find_cursor_binary() {
                let mut meta = serde_json::Map::new();
                meta.insert(
                    "terminal-auth".to_string(),
                    serde_json::json!({
                        "command": cli_path.to_string_lossy(),
                        "args": ["login"]
                    }),
                );
                login_method = login_method.meta(meta);
            }
        }

        methods.push(login_method);
        methods.push(
            AuthMethod::new(CursorAuthMethod::ApiKey, "Use API Key")
                .description("Set the CURSOR_API_KEY environment variable"),
        );

        methods
    }

    /// Build available slash commands
    fn build_available_commands(&self) -> Vec<AvailableCommand> {
        vec![AvailableCommand::new(
            "login",
            "Authenticate with Cursor via browser login",
        )]
    }

    /// Send available commands update to the client
    fn send_available_commands(&self, session_id: SessionId) {
        let commands = self.build_available_commands();
        tokio::task::spawn_local(async move {
            if let Some(client) = ACP_CLIENT.get() {
                let update =
                    SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(commands));
                let notification = SessionNotification::new(session_id, update);
                if let Err(e) = client.session_notification(notification).await {
                    tracing::error!("Failed to send available commands: {:?}", e);
                }
            }
        });
    }
}

impl Default for CursorAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait(?Send)]
impl Agent for CursorAgent {
    async fn initialize(&self, request: InitializeRequest) -> Result<InitializeResponse, Error> {
        let InitializeRequest {
            protocol_version,
            client_capabilities,
            client_info,
            ..
        } = request;

        if let Some(info) = &client_info {
            debug!(
                "Initialize request from client: {} v{} (protocol: {:?})",
                info.name, info.version, protocol_version
            );
        } else {
            debug!("Initialize request (protocol: {:?})", protocol_version);
        }

        *self.client_capabilities.lock().unwrap() = client_capabilities;

        let session_capabilities = SessionCapabilities::new().list(SessionListCapabilities::new());

        let mut agent_capabilities = AgentCapabilities::new()
            .prompt_capabilities(PromptCapabilities::new().embedded_context(true).image(true))
            .load_session(true);
        agent_capabilities.session_capabilities = session_capabilities;

        debug!(
            "Agent capabilities being sent: {:?}",
            serde_json::to_string(&agent_capabilities)
        );

        let auth_methods = self.build_auth_methods();

        Ok(InitializeResponse::new(ProtocolVersion::V1)
            .agent_capabilities(agent_capabilities)
            .agent_info(
                Implementation::new("cursor-acp", env!("CARGO_PKG_VERSION")).title("Cursor"),
            )
            .auth_methods(auth_methods))
    }

    async fn authenticate(
        &self,
        request: AuthenticateRequest,
    ) -> Result<AuthenticateResponse, Error> {
        let auth_method = CursorAuthMethod::try_from(request.method_id)?;

        match auth_method {
            CursorAuthMethod::BrowserLogin => {
                info!("Starting browser login flow");

                let binary_path = find_cursor_binary().map_err(|e| {
                    Error::internal_error().data(format!("Cursor CLI not found: {e}"))
                })?;

                info!("Launching login with binary: {:?}", binary_path);

                let result = tokio::process::Command::new(&binary_path)
                    .arg("login")
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .status()
                    .await;

                match result {
                    Ok(status) if status.success() => {
                        info!("Browser login flow completed successfully");
                        *self.authenticated.borrow_mut() = true;
                    }
                    Ok(status) => {
                        info!("Login process exited with status: {}", status);
                        // Still mark as authenticated - user may have completed login
                        *self.authenticated.borrow_mut() = true;
                    }
                    Err(e) => {
                        return Err(
                            Error::internal_error().data(format!("Failed to launch login: {e}"))
                        );
                    }
                }
            }
            CursorAuthMethod::ApiKey => {
                if std::env::var("CURSOR_API_KEY").is_err() {
                    return Err(
                        Error::internal_error().data("CURSOR_API_KEY environment variable not set")
                    );
                }
                info!("API key authentication successful");
                *self.authenticated.borrow_mut() = true;
            }
        }

        Ok(AuthenticateResponse::new())
    }

    async fn new_session(&self, request: NewSessionRequest) -> Result<NewSessionResponse, Error> {
        // Don't require auth upfront - Cursor CLI will prompt if needed
        // and we'll get an error we can surface

        let NewSessionRequest { cwd, .. } = request;
        info!("Creating new session with cwd: {}", cwd.display());

        let session_id = self.next_session_id();

        // Record the session root
        self.session_roots
            .lock()
            .unwrap()
            .insert(session_id.clone(), cwd.clone());

        let (selected_mode, selected_model) = {
            let storage = self.session_storage.borrow();
            let mode = match storage.last_mode.as_deref() {
                Some("plan") => "plan",
                Some("ask") => "ask",
                _ => "default",
            };
            let model = match storage.last_model.as_deref() {
                Some("opus-4.5-thinking") => "opus-4.5-thinking",
                Some("opus-4.5") => "opus-4.5",
                Some("sonnet-4.5-thinking") => "sonnet-4.5-thinking",
                Some("sonnet-4.5") => "sonnet-4.5",
                Some("gpt-5.2") => "gpt-5.2",
                Some("gemini-3-pro") => "gemini-3-pro",
                _ => "auto",
            };
            (mode, model)
        };

        let session = Rc::new(Session::new(
            session_id.clone(),
            cwd.clone(),
            self.client_capabilities.clone(),
        ));

        if selected_model != "auto" {
            session.set_model(selected_model.to_string());
        }
        if selected_mode != "default" {
            session.set_mode(selected_mode.to_string());
        }

        self.sessions
            .borrow_mut()
            .insert(session_id.clone(), session);

        // Persist session metadata
        let metadata = SessionMetadata::new(session_id.clone(), cwd);
        self.session_storage.borrow_mut().upsert(metadata);

        debug!("Created new session: {}", session_id.0);

        self.send_available_commands(session_id.clone());

        let modes = SessionModeState::new(
            selected_mode,
            vec![
                SessionMode::new("default", "Default")
                    .description("Normal mode with full tool access"),
                SessionMode::new("plan", "Plan").description("Read-only planning mode"),
                SessionMode::new("ask", "Ask").description("Q&A mode for explanations"),
            ],
        );

        let models = SessionModelState::new(
            selected_model,
            vec![
                ModelInfo::new("auto", "Auto"),
                ModelInfo::new("opus-4.5-thinking", "Claude 4.5 Opus (Thinking)"),
                ModelInfo::new("opus-4.5", "Claude 4.5 Opus"),
                ModelInfo::new("sonnet-4.5-thinking", "Claude 4.5 Sonnet (Thinking)"),
                ModelInfo::new("sonnet-4.5", "Claude 4.5 Sonnet"),
                ModelInfo::new("gpt-5.2", "GPT-5.2"),
                ModelInfo::new("gemini-3-pro", "Gemini 3 Pro"),
            ],
        );

        Ok(NewSessionResponse::new(session_id)
            .modes(modes)
            .models(models))
    }

    async fn load_session(
        &self,
        request: LoadSessionRequest,
    ) -> Result<LoadSessionResponse, Error> {
        let session_id_str = request.session_id.0.to_string();
        info!("Load session request: {}", session_id_str);

        // Check if session exists in storage
        let metadata = {
            let storage = self.session_storage.borrow();
            storage.get(&session_id_str).cloned()
        };

        let Some(metadata) = metadata else {
            info!("Session {} not found in storage", session_id_str);
            return Err(Error::resource_not_found(None));
        };

        // Create a new session with the stored CWD
        let session = Rc::new(Session::new(
            request.session_id.clone(),
            metadata.cwd.clone(),
            self.client_capabilities.clone(),
        ));

        // Restore Cursor CLI session ID for conversation continuity
        if let Some(cursor_sid) = &metadata.cursor_session_id {
            session.set_cursor_session_id(cursor_sid.clone());
            debug!("Restored Cursor session ID: {}", cursor_sid);
        }

        {
            let storage = self.session_storage.borrow();
            if let Some(model) = &storage.last_model {
                session.set_model(model.clone());
            }
            if let Some(mode) = &storage.last_mode {
                session.set_mode(mode.clone());
            }
        }

        // Store the session
        self.sessions
            .borrow_mut()
            .insert(request.session_id.clone(), session);

        self.session_roots
            .lock()
            .unwrap()
            .insert(request.session_id.clone(), metadata.cwd.clone());

        // Touch the session to update last access time
        self.session_storage.borrow_mut().touch(&session_id_str);

        debug!("Loaded session: {}", session_id_str);

        self.send_available_commands(request.session_id.clone());

        let (selected_mode, selected_model) = {
            let storage = self.session_storage.borrow();
            let mode = match storage.last_mode.as_deref() {
                Some("plan") => "plan",
                Some("ask") => "ask",
                _ => "default",
            };
            let model = match storage.last_model.as_deref() {
                Some("opus-4.5-thinking") => "opus-4.5-thinking",
                Some("opus-4.5") => "opus-4.5",
                Some("sonnet-4.5-thinking") => "sonnet-4.5-thinking",
                Some("sonnet-4.5") => "sonnet-4.5",
                Some("gpt-5.2") => "gpt-5.2",
                Some("gemini-3-pro") => "gemini-3-pro",
                _ => "auto",
            };
            (mode, model)
        };

        let modes = SessionModeState::new(
            selected_mode,
            vec![
                SessionMode::new("default", "Default")
                    .description("Normal mode with full tool access"),
                SessionMode::new("plan", "Plan").description("Read-only planning mode"),
                SessionMode::new("ask", "Ask").description("Q&A mode for explanations"),
            ],
        );

        let models = SessionModelState::new(
            selected_model,
            vec![
                ModelInfo::new("auto", "Auto"),
                ModelInfo::new("opus-4.5-thinking", "Claude 4.5 Opus (Thinking)"),
                ModelInfo::new("opus-4.5", "Claude 4.5 Opus"),
                ModelInfo::new("sonnet-4.5-thinking", "Claude 4.5 Sonnet (Thinking)"),
                ModelInfo::new("sonnet-4.5", "Claude 4.5 Sonnet"),
                ModelInfo::new("gpt-5.2", "GPT-5.2"),
                ModelInfo::new("gemini-3-pro", "Gemini 3 Pro"),
            ],
        );

        Ok(LoadSessionResponse::new().modes(modes).models(models))
    }

    async fn list_sessions(
        &self,
        _request: ListSessionsRequest,
    ) -> Result<ListSessionsResponse, Error> {
        let storage = self.session_storage.borrow();
        let sessions: Vec<SessionInfo> = storage
            .list()
            .into_iter()
            .map(|meta| {
                let mut info =
                    SessionInfo::new(SessionId::new(meta.session_id.clone()), meta.cwd.clone());

                if let Some(title) = &meta.title {
                    info = info.title(title.clone());
                }

                // Set the updated_at timestamp
                info = info.updated_at(meta.updated_at.to_rfc3339());

                info
            })
            .collect();

        info!("Listing {} sessions:", sessions.len());
        for session in &sessions {
            debug!(
                "  - {} (title: {:?}, cwd: {})",
                session.session_id.0,
                session.title,
                session.cwd.display()
            );
        }
        Ok(ListSessionsResponse::new(sessions))
    }

    async fn prompt(&self, request: PromptRequest) -> Result<PromptResponse, Error> {
        let session_id_str = request.session_id.0.to_string();
        info!("Processing prompt for session: {}", session_id_str);

        // Extract prompt text for title generation
        let prompt_text: String = request
            .prompt
            .iter()
            .filter_map(|block| {
                if let agent_client_protocol::ContentBlock::Text(t) = block {
                    Some(t.text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Generate title from first prompt and update session
        if !prompt_text.is_empty() {
            let title = generate_title(&prompt_text);
            self.session_storage
                .borrow_mut()
                .set_title(&session_id_str, title);
        }

        // Touch the session to update last activity time
        self.session_storage.borrow_mut().touch(&session_id_str);

        let session = self.get_session(&request.session_id)?;
        let result = session.prompt(request).await;

        // Save the Cursor CLI session ID (even on error)
        if let Some(cursor_sid) = session.cursor_session_id() {
            self.session_storage
                .borrow_mut()
                .set_cursor_session_id(&session_id_str, cursor_sid);
        }

        Ok(PromptResponse::new(result?))
    }

    async fn cancel(&self, notification: CancelNotification) -> Result<(), Error> {
        info!(
            "Cancelling operations for session: {}",
            notification.session_id
        );
        if let Ok(session) = self.get_session(&notification.session_id) {
            session.cancel().await;
        }
        Ok(())
    }

    async fn set_session_mode(
        &self,
        request: SetSessionModeRequest,
    ) -> Result<SetSessionModeResponse, Error> {
        info!("Setting mode to {:?}", request.mode_id);

        if let Ok(session) = self.get_session(&request.session_id) {
            // Map ACP mode IDs to Cursor CLI modes
            let cursor_mode = match request.mode_id.0.as_ref() {
                "plan" => Some("plan"),
                "ask" => Some("ask"),
                "default" | "normal" => None, // No --mode flag for default
                _ => None,
            };
            if let Some(mode) = cursor_mode {
                session.set_mode(mode.to_string());
            }
        }

        self.session_storage
            .borrow_mut()
            .set_mode(request.mode_id.0.to_string());

        Ok(SetSessionModeResponse::default())
    }

    async fn set_session_model(
        &self,
        request: SetSessionModelRequest,
    ) -> Result<SetSessionModelResponse, Error> {
        info!("Setting model to {:?}", request.model_id);

        if let Ok(session) = self.get_session(&request.session_id) {
            session.set_model(request.model_id.0.to_string());
        }

        self.session_storage
            .borrow_mut()
            .set_model(request.model_id.0.to_string());

        Ok(SetSessionModelResponse::default())
    }

    async fn set_session_config_option(
        &self,
        request: SetSessionConfigOptionRequest,
    ) -> Result<SetSessionConfigOptionResponse, Error> {
        info!(
            "Setting session config option for session: {} (config_id: {}, value: {})",
            request.session_id, request.config_id.0, request.value.0
        );
        // Not implemented yet
        Ok(SetSessionConfigOptionResponse::new(vec![]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_capabilities_serialization() {
        let session_capabilities = SessionCapabilities::new().list(SessionListCapabilities::new());
        let mut agent_capabilities = AgentCapabilities::new()
            .prompt_capabilities(PromptCapabilities::new().embedded_context(true).image(true))
            .load_session(true);
        agent_capabilities.session_capabilities = session_capabilities.clone();

        let json = serde_json::to_string_pretty(&agent_capabilities).unwrap();
        println!("Agent capabilities JSON:\n{}", json);

        // Verify that session_capabilities.list is present
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            value["sessionCapabilities"]["list"].is_object(),
            "sessionCapabilities.list should be present as an object"
        );

        // Test the full InitializeResponse
        let auth_methods = vec![AuthMethod::new(
            CursorAuthMethod::BrowserLogin,
            "Login with Cursor",
        )];

        let response = InitializeResponse::new(ProtocolVersion::V1)
            .agent_capabilities(agent_capabilities)
            .agent_info(
                Implementation::new("cursor-acp", env!("CARGO_PKG_VERSION")).title("Cursor"),
            )
            .auth_methods(auth_methods);

        let response_json = serde_json::to_string_pretty(&response).unwrap();
        println!("\nFull InitializeResponse JSON:\n{}", response_json);

        let response_value: serde_json::Value = serde_json::from_str(&response_json).unwrap();
        assert!(
            response_value["agentCapabilities"]["sessionCapabilities"]["list"].is_object(),
            "Full response should have agentCapabilities.sessionCapabilities.list"
        );
    }
}
