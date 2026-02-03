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

use crate::session::Session;

/// Authentication method for Cursor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorAuthMethod {
    /// Use CURSOR_API_KEY environment variable
    CursorApiKey,
}

impl From<CursorAuthMethod> for AuthMethodId {
    fn from(method: CursorAuthMethod) -> Self {
        Self::new(match method {
            CursorAuthMethod::CursorApiKey => "cursor-api-key",
        })
    }
}

impl From<CursorAuthMethod> for AuthMethod {
    fn from(method: CursorAuthMethod) -> Self {
        match method {
            CursorAuthMethod::CursorApiKey => Self::new(method, "Use CURSOR_API_KEY")
                .description("Requires setting the `CURSOR_API_KEY` environment variable."),
        }
    }
}

impl TryFrom<AuthMethodId> for CursorAuthMethod {
    type Error = Error;

    fn try_from(value: AuthMethodId) -> Result<Self, Self::Error> {
        match value.0.as_ref() {
            "cursor-api-key" => Ok(CursorAuthMethod::CursorApiKey),
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
}

impl CursorAgent {
    /// Create a new Cursor agent
    pub fn new() -> Self {
        Self {
            client_capabilities: Arc::default(),
            sessions: Rc::default(),
            session_roots: Arc::default(),
            next_session_id: RefCell::new(1),
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

    fn check_auth(&self) -> Result<(), Error> {
        // Check if CURSOR_API_KEY is set
        if std::env::var("CURSOR_API_KEY").is_err() {
            return Err(Error::auth_required());
        }
        Ok(())
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

        let auth_methods = vec![CursorAuthMethod::CursorApiKey.into()];

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
            CursorAuthMethod::CursorApiKey => {
                if std::env::var("CURSOR_API_KEY").is_err() {
                    return Err(
                        Error::internal_error().data("CURSOR_API_KEY environment variable not set")
                    );
                }
            }
        }

        info!("Authentication successful via {:?}", auth_method);
        Ok(AuthenticateResponse::new())
    }

    async fn new_session(&self, request: NewSessionRequest) -> Result<NewSessionResponse, Error> {
        self.check_auth()?;

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
        self.check_auth()?;

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
