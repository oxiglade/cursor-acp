//! Cursor ACP - An Agent Client Protocol adapter for Cursor CLI.

use agent_client_protocol::AgentSideConnection;
use std::sync::{Arc, OnceLock};
use std::{io::Result as IoResult, rc::Rc};
use tokio::task::LocalSet;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use tracing_subscriber::EnvFilter;

mod cursor_agent;
mod cursor_process;
mod session;
mod session_storage;
mod stream_json;

pub use cursor_agent::CursorAgent;

/// Global ACP client connection for sending notifications
pub static ACP_CLIENT: OnceLock<Arc<AgentSideConnection>> = OnceLock::new();

/// Run the Cursor ACP agent.
///
/// This sets up an ACP agent that communicates over stdio, bridging
/// the ACP protocol with the Cursor CLI.
///
/// # Errors
///
/// If unable to start the agent or communicate over stdio.
pub async fn run_main() -> IoResult<()> {
    // Install a simple subscriber so `tracing` output is visible.
    // Users can control the log level with `RUST_LOG`.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    tracing::info!("Starting cursor-acp adapter v{}", env!("CARGO_PKG_VERSION"));

    let agent = Rc::new(CursorAgent::new());

    let stdin = tokio::io::stdin().compat();
    let stdout = tokio::io::stdout().compat_write();

    LocalSet::new()
        .run_until(async move {
            let (client, io_task) = AgentSideConnection::new(agent.clone(), stdout, stdin, |fut| {
                tokio::task::spawn_local(fut);
            });

            if ACP_CLIENT.set(Arc::new(client)).is_err() {
                return Err(std::io::Error::other("ACP client already set"));
            }

            io_task
                .await
                .map_err(|e| std::io::Error::other(format!("ACP I/O error: {e}")))
        })
        .await?;

    Ok(())
}
