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

/// Run a Cursor CLI command with the given args (e.g. `["--version"]`, `["models"]`, `["create-chat"]`).
/// Returns (success, stdout, stderr). Fails if the binary cannot be found or the process fails to start.
pub async fn run_cursor_command(args: &[&str]) -> Result<(bool, String, String)> {
    let cursor_bin = find_cursor_binary()?;
    let output = Command::new(&cursor_bin)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("failed to run Cursor CLI")?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    Ok((output.status.success(), stdout, stderr))
}

/// Returns true if the Cursor CLI is available (binary found and `--version` succeeds).
/// Does not fail; returns false on any error.
pub async fn check_cursor_cli_available() -> bool {
    if find_cursor_binary().is_err() {
        return false;
    }
    run_cursor_command(&["--version"])
        .await
        .map(|(ok, _, _)| ok)
        .unwrap_or(false)
}

/// Create a new Cursor chat and return its ID. Pass to spawn as `--resume <id>`.
pub async fn create_cursor_chat() -> Result<String> {
    let (ok, stdout, stderr) = run_cursor_command(&["create-chat"]).await?;
    let id = stdout.trim().to_string();
    if !ok {
        anyhow::bail!("cursor-agent create-chat failed: {}", stderr.trim());
    }
    if id.is_empty() {
        anyhow::bail!("cursor-agent create-chat returned empty ID");
    }
    Ok(id)
}

/// Parse the output of `cursor-agent models` into (model_id, display_name) pairs.
/// Expects lines like "model-id - Display Name" or "model-id - Display Name  (current)" after an "Available models" header.
pub fn parse_models_output(stdout: &str) -> Vec<(String, String)> {
    let mut models = Vec::new();
    let mut in_models_section = false;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.to_lowercase().contains("available models") {
            in_models_section = true;
            continue;
        }
        if trimmed.to_lowercase().starts_with("tip:") {
            continue;
        }
        if in_models_section {
            let dash = " - ";
            if let Some(pos) = trimmed.find(dash) {
                let id = trimmed[..pos].trim().to_string();
                let mut name = trimmed[pos + dash.len()..].to_string();
                // Remove "(current)" or "(default)" suffix
                if let Some(open) = name.find('(') {
                    name = name[..open].trim().to_string();
                }
                let name = name.trim().to_string();
                if !id.is_empty() && !name.is_empty() {
                    // Avoid duplicate "auto"
                    if id == "auto" && models.iter().any(|(i, _)| i == "auto") {
                        continue;
                    }
                    models.push((id, name));
                }
            }
        }
    }
    // Ensure "auto" is first if present
    if let Some(pos) = models.iter().position(|(id, _)| id == "auto") {
        if pos > 0 {
            let auto = models.remove(pos);
            models.insert(0, auto);
        }
    } else if !models.is_empty() {
        models.insert(0, ("auto".to_string(), "Auto".to_string()));
    }
    models
}

/// Load the list of models from the Cursor CLI. On failure returns an error; caller can fall back to a default list.
pub async fn list_cursor_models() -> Result<Vec<(String, String)>> {
    let (ok, stdout, _) = run_cursor_command(&["models"]).await?;
    if !ok {
        anyhow::bail!("cursor-agent models failed");
    }
    let models = parse_models_output(&stdout);
    if models.is_empty() {
        anyhow::bail!("cursor-agent models produced no models");
    }
    Ok(models)
}

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

        // Pass --resume with session ID when resuming
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

#[cfg(test)]
mod tests {
    use super::parse_models_output;

    #[test]
    fn test_parse_models_output() {
        let output = r#"
Available models

auto - Auto  (current)
composer-1 - Composer 1
opus-4.5 - Claude 4.5 Opus
"#;
        let models = parse_models_output(output);
        assert!(!models.is_empty());
        assert_eq!(models[0], ("auto".to_string(), "Auto".to_string()));
        assert!(models.iter().any(|(id, _)| id == "composer-1"));
        assert!(models.iter().any(|(id, _)| id == "opus-4.5"));
    }

    #[test]
    fn test_parse_models_output_empty_after_header() {
        let output = "Available models\n\n";
        let models = parse_models_output(output);
        assert!(models.is_empty());
    }

    #[test]
    fn test_parse_models_output_strips_default_suffix() {
        let output = "Available models\n\nsonnet-4.5 - Claude 4.5 Sonnet  (default)\n";
        let models = parse_models_output(output);
        assert!(models
            .iter()
            .any(|(id, name)| id == "sonnet-4.5" && name == "Claude 4.5 Sonnet"));
    }

    #[test]
    fn test_parse_models_output_skips_tip_lines() {
        let output = "Available models\n\ntip: run cursor-agent models\nauto - Auto\n";
        let models = parse_models_output(output);
        assert!(models.iter().any(|(id, _)| id == "auto"));
        assert!(!models.iter().any(|(id, _)| id == "tip:"));
    }
}
