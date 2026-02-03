//! Cursor CLI process management.
//!
//! This module handles spawning and communicating with the Cursor CLI
//! in headless mode using stream-json output format.

use std::path::PathBuf;
use std::process::Stdio;

use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

use crate::stream_json::StreamEvent;

/// Finds the Cursor CLI binary
pub fn find_cursor_binary() -> Result<PathBuf> {
    // The CLI is called "agent" or "cursor-agent"
    // Check PATH first
    for name in ["agent", "cursor-agent"] {
        if let Ok(path) = which::which(name) {
            return Ok(path);
        }
    }

    // Check common installation locations
    let home = std::env::var("HOME").context("HOME not set")?;

    let candidates = [
        // Standard install location (from curl installer)
        format!("{home}/.local/bin/agent"),
        format!("{home}/.local/bin/cursor-agent"),
        // macOS alternatives
        "/usr/local/bin/agent".to_string(),
        "/usr/local/bin/cursor-agent".to_string(),
        "/opt/homebrew/bin/agent".to_string(),
        "/opt/homebrew/bin/cursor-agent".to_string(),
        // Linux alternatives
        "/usr/bin/agent".to_string(),
        "/usr/bin/cursor-agent".to_string(),
    ];

    for candidate in candidates {
        let path = PathBuf::from(&candidate);
        if path.exists() {
            return Ok(path);
        }
    }

    anyhow::bail!(
        "Could not find Cursor CLI (agent/cursor-agent). Please install it:\n\
         curl https://cursor.com/install -fsSL | bash"
    )
}

/// A handle to a running Cursor CLI process
pub struct CursorProcess {
    child: Option<Child>,
    event_rx: Option<mpsc::Receiver<StreamEvent>>,
}

/// Separated components of a spawned process.
pub struct CursorProcessParts {
    pub child: Child,
    pub event_rx: mpsc::Receiver<StreamEvent>,
}

impl CursorProcess {
    /// Spawn a new Cursor CLI process with the given prompt
    pub async fn spawn(
        prompt: &str,
        working_dir: Option<&std::path::Path>,
        model: Option<&str>,
        mode: Option<&str>,
        resume_session_id: Option<&str>,
    ) -> Result<Self> {
        let cursor_bin = find_cursor_binary()?;

        let mut cmd = Command::new(&cursor_bin);

        // Headless mode with stream-json output (flags first, then prompt)
        cmd.arg("-p").arg("--output-format").arg("stream-json");

        // Skip permission prompts - the ACP client handles permissions
        cmd.arg("--force");

        if let Some(model) = model {
            cmd.arg("--model").arg(model);
        }

        if let Some(mode) = mode {
            cmd.arg("--mode").arg(mode);
        }

        // Resume previous session for conversation continuity
        if let Some(session_id) = resume_session_id {
            cmd.arg("--resume").arg(session_id);
            tracing::info!("Resuming Cursor session: {}", session_id);
        } else {
            tracing::info!("Starting fresh Cursor session (no resume ID)");
        }

        // Prompt comes last
        cmd.arg(prompt);

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        tracing::debug!("Spawning Cursor CLI: {:?}", cmd);

        let mut child = cmd.spawn().context("failed to spawn Cursor CLI")?;

        // Set up stdout reader
        let stdout = child.stdout.take().context("no stdout")?;
        let stderr = child.stderr.take().context("no stderr")?;
        let (event_tx, event_rx) = mpsc::channel(100);

        // Spawn task to read stderr and log it
        tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::warn!("CLI stderr: {}", line);
            }
        });

        // Spawn task to read stdout and parse events
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();

            tracing::debug!("Starting to read CLI output");
            while let Ok(Some(line)) = lines.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }

                tracing::debug!("Received line from CLI: {}", &line[..line.len().min(200)]);
                match StreamEvent::parse(&line) {
                    Ok(event) => {
                        tracing::debug!("Parsed event: {:?}", event);
                        if event_tx.send(event).await.is_err() {
                            tracing::debug!("Event channel closed");
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse stream event: {e}\nLine: {line}");
                    }
                }
            }
            tracing::debug!("CLI output stream ended");
        });

        Ok(Self {
            child: Some(child),
            event_rx: Some(event_rx),
        })
    }

    /// Consume the process, returning the child handle and event receiver separately.
    pub fn into_parts(mut self) -> CursorProcessParts {
        CursorProcessParts {
            child: self.child.take().expect("into_parts called more than once"),
            event_rx: self
                .event_rx
                .take()
                .expect("into_parts called more than once"),
        }
    }

    /// Receive the next event from the Cursor CLI
    #[allow(dead_code)]
    pub async fn next_event(&mut self) -> Option<StreamEvent> {
        self.event_rx.as_mut()?.recv().await
    }

    /// Check if the process is still running
    #[allow(dead_code)]
    pub fn is_running(&mut self) -> bool {
        self.child
            .as_mut()
            .is_some_and(|c| matches!(c.try_wait(), Ok(None)))
    }

    /// Kill the process
    #[allow(dead_code)]
    pub async fn kill(&mut self) -> Result<()> {
        if let Some(ref mut child) = self.child {
            child.kill().await.context("failed to kill process")
        } else {
            Ok(())
        }
    }

    /// Wait for the process to exit
    #[allow(dead_code)]
    pub async fn wait(&mut self) -> Result<std::process::ExitStatus> {
        if let Some(ref mut child) = self.child {
            child.wait().await.context("failed to wait for process")
        } else {
            anyhow::bail!("process already taken via into_parts");
        }
    }
}

impl Drop for CursorProcess {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.child {
            drop(child.start_kill());
        }
    }
}
