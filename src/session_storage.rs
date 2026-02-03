//! Session persistence for Cursor ACP.
//!
//! This module handles saving and loading session metadata to disk,
//! enabling session history to survive restarts.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use agent_client_protocol::SessionId;

/// Maximum number of sessions to keep in history
const MAX_SESSIONS: usize = 50;

/// Metadata about a saved session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// The unique session ID
    pub session_id: String,
    /// Human-readable title (derived from first prompt)
    pub title: Option<String>,
    /// Working directory for this session
    pub cwd: PathBuf,
    /// When the session was created
    pub created_at: DateTime<Utc>,
    /// When the session was last active
    pub updated_at: DateTime<Utc>,
    /// Cursor CLI's internal session ID for conversation continuity
    #[serde(default)]
    pub cursor_session_id: Option<String>,
    /// Selected model for this session
    #[serde(default)]
    pub model: Option<String>,
    /// Selected mode for this session
    #[serde(default)]
    pub mode: Option<String>,
}

impl SessionMetadata {
    /// Create new session metadata
    pub fn new(session_id: SessionId, cwd: PathBuf) -> Self {
        let now = Utc::now();
        Self {
            session_id: session_id.0.to_string(),
            title: None,
            cwd,
            created_at: now,
            updated_at: now,
            cursor_session_id: None,
            model: None,
            mode: None,
        }
    }

    /// Update the title from the first prompt
    pub fn set_title(&mut self, title: String) {
        if self.title.is_none() {
            self.title = Some(title);
        }
    }

    /// Mark the session as recently active
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }
}

/// Storage for session history
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SessionStorage {
    /// Map of session ID to metadata
    sessions: HashMap<String, SessionMetadata>,
}

impl SessionStorage {
    /// Get the storage file path
    fn storage_path() -> Option<PathBuf> {
        dirs::data_local_dir().map(|p| p.join("cursor-acp").join("sessions.json"))
    }

    /// Load session storage from disk
    pub fn load() -> Self {
        let Some(path) = Self::storage_path() else {
            tracing::warn!("Could not determine data directory");
            return Self::default();
        };

        if !path.exists() {
            return Self::default();
        }

        match fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(storage) => {
                    tracing::debug!("Loaded session storage from {:?}", path);
                    storage
                }
                Err(e) => {
                    tracing::warn!("Failed to parse session storage: {}", e);
                    Self::default()
                }
            },
            Err(e) => {
                tracing::warn!("Failed to read session storage: {}", e);
                Self::default()
            }
        }
    }

    /// Save session storage to disk
    pub fn save(&self) {
        let Some(path) = Self::storage_path() else {
            tracing::warn!("Could not determine data directory");
            return;
        };

        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                tracing::warn!("Failed to create data directory: {}", e);
                return;
            }
        }

        match serde_json::to_string_pretty(self) {
            Ok(contents) => {
                if let Err(e) = fs::write(&path, contents) {
                    tracing::warn!("Failed to write session storage: {}", e);
                } else {
                    tracing::debug!("Saved session storage to {:?}", path);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to serialize session storage: {}", e);
            }
        }
    }

    /// Add or update a session
    pub fn upsert(&mut self, metadata: SessionMetadata) {
        self.sessions.insert(metadata.session_id.clone(), metadata);
        self.prune();
        self.save();
    }

    /// Update a session's title
    pub fn set_title(&mut self, session_id: &str, title: String) {
        if let Some(meta) = self.sessions.get_mut(session_id) {
            meta.set_title(title);
            self.save();
        }
    }

    /// Update a session's Cursor CLI session ID
    pub fn set_cursor_session_id(&mut self, session_id: &str, cursor_session_id: String) {
        if let Some(meta) = self.sessions.get_mut(session_id) {
            meta.cursor_session_id = Some(cursor_session_id);
            self.save();
        }
    }

    /// Update a session's model
    pub fn set_model(&mut self, session_id: &str, model: String) {
        if let Some(meta) = self.sessions.get_mut(session_id) {
            meta.model = Some(model);
            self.save();
        }
    }

    /// Update a session's mode
    pub fn set_mode(&mut self, session_id: &str, mode: String) {
        if let Some(meta) = self.sessions.get_mut(session_id) {
            meta.mode = Some(mode);
            self.save();
        }
    }

    /// Mark a session as recently active
    pub fn touch(&mut self, session_id: &str) {
        if let Some(meta) = self.sessions.get_mut(session_id) {
            meta.touch();
            self.save();
        }
    }

    /// Get a session by ID
    pub fn get(&self, session_id: &str) -> Option<&SessionMetadata> {
        self.sessions.get(session_id)
    }

    /// List all sessions, sorted by most recently updated
    pub fn list(&self) -> Vec<&SessionMetadata> {
        let mut sessions: Vec<_> = self.sessions.values().collect();
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        sessions
    }

    /// Remove old sessions to stay under the limit
    fn prune(&mut self) {
        if self.sessions.len() <= MAX_SESSIONS {
            return;
        }

        // Sort by updated_at and keep the newest
        let mut sessions: Vec<_> = self.sessions.drain().collect();
        sessions.sort_by(|a, b| b.1.updated_at.cmp(&a.1.updated_at));
        sessions.truncate(MAX_SESSIONS);

        self.sessions = sessions.into_iter().collect();
    }

    /// Delete a session from storage
    #[allow(dead_code)]
    pub fn delete(&mut self, session_id: &str) {
        self.sessions.remove(session_id);
        self.save();
    }
}

/// Generate a title from a prompt string
pub fn generate_title(prompt: &str) -> String {
    // Take the first line or first 60 characters
    let first_line = prompt.lines().next().unwrap_or(prompt);

    // Clean up and truncate
    let cleaned = first_line.trim();

    if cleaned.len() <= 60 {
        cleaned.to_string()
    } else {
        // Find a word boundary near 60 chars
        let truncated = &cleaned[..60];
        if let Some(last_space) = truncated.rfind(' ') {
            format!("{}...", &truncated[..last_space])
        } else {
            format!("{}...", truncated)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_title_short() {
        assert_eq!(generate_title("Fix the bug"), "Fix the bug");
    }

    #[test]
    fn test_generate_title_long() {
        let long_prompt = "This is a very long prompt that should be truncated because it exceeds the maximum length allowed for titles";
        let title = generate_title(long_prompt);
        assert!(title.len() <= 63); // 60 + "..."
        assert!(title.ends_with("..."));
    }

    #[test]
    fn test_generate_title_multiline() {
        let multiline = "First line of prompt\nSecond line\nThird line";
        assert_eq!(generate_title(multiline), "First line of prompt");
    }
}
