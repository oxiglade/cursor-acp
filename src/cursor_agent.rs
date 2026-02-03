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
    CancelNotification, ClientCapabilities, Error, Implementation, InitializeRequest,
    InitializeResponse, ListSessionsRequest, ListSessionsResponse, LoadSessionRequest,
    LoadSessionResponse, NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest,
    PromptResponse, ProtocolVersion, SessionId, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, SetSessionModeRequest, SetSessionModeResponse,
    SetSessionModelRequest, SetSessionModelResponse,
};
use tracing::{debug, info};

use crate::cursor_process::find_cursor_binary;
use crate::session::Session;

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
}

impl CursorAgent {
    /// Create a new Cursor agent
    pub fn new() -> Self {
        // Check if already authenticated (has API key or cached credentials)
        let authenticated = std::env::var("CURSOR_API_KEY").is_ok();

        Self {
            client_capabilities: Arc::default(),
            sessions: Rc::default(),
            session_roots: Arc::default(),
            next_session_id: RefCell::new(1),
            authenticated: RefCell::new(authenticated),
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

        // Primary: Browser login (if terminal-auth is supported or as default)
        let mut login_method =
            AuthMethod::new(CursorAuthMethod::BrowserLogin, "Login with Cursor").description(
                "Opens a browser to authenticate with your Cursor account",
            );

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

        // Secondary: API key
        methods.push(
            AuthMethod::new(CursorAuthMethod::ApiKey, "Use API Key")
                .description("Set the CURSOR_API_KEY environment variable"),
        );

        methods
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

        let agent_capabilities = AgentCapabilities::new()
            .prompt_capabilities(PromptCapabilities::new().embedded_context(true).image(true))
            .load_session(false); // We don't persist sessions yet

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
                // The client should have launched `cursor-agent login` via terminal-auth
                // or the user ran it manually. We assume auth is complete after this.
                info!("Browser login flow completed");
                *self.authenticated.borrow_mut() = true;
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

        let session = Rc::new(Session::new(
            session_id.clone(),
            cwd,
            self.client_capabilities.clone(),
        ));

        self.sessions
            .borrow_mut()
            .insert(session_id.clone(), session);

        debug!("Created new session: {}", session_id.0);

        Ok(NewSessionResponse::new(session_id))
    }

    async fn load_session(
        &self,
        request: LoadSessionRequest,
    ) -> Result<LoadSessionResponse, Error> {
        // We don't persist sessions yet
        info!(
            "Load session request: {} (not supported)",
            request.session_id
        );
        Err(Error::resource_not_found(None))
    }

    async fn list_sessions(
        &self,
        _request: ListSessionsRequest,
    ) -> Result<ListSessionsResponse, Error> {
        // Return empty list - we don't persist sessions
        Ok(ListSessionsResponse::new(vec![]))
    }

    async fn prompt(&self, request: PromptRequest) -> Result<PromptResponse, Error> {
        info!("Processing prompt for session: {}", request.session_id);

        let session = self.get_session(&request.session_id)?;
        let stop_reason = session.prompt(request).await?;

        Ok(PromptResponse::new(stop_reason))
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
        info!(
            "Setting session mode for session: {} to {:?}",
            request.session_id, request.mode_id
        );
        // Cursor doesn't have explicit modes in CLI
        Ok(SetSessionModeResponse::default())
    }

    async fn set_session_model(
        &self,
        request: SetSessionModelRequest,
    ) -> Result<SetSessionModelResponse, Error> {
        info!(
            "Setting session model for session: {} to {:?}",
            request.session_id, request.model_id
        );

        if let Ok(session) = self.get_session(&request.session_id) {
            session.set_model(request.model_id.0.to_string());
        }

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
