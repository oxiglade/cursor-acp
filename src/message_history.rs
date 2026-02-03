//! Message history storage for chat sessions.
//!
//! This module handles persisting conversation messages to disk,
//! enabling full chat history replay when sessions are loaded.

use std::fs;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single message in the conversation history
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HistoryMessage {
    /// A user prompt
    User {
        content: String,
        timestamp: DateTime<Utc>,
    },
    /// Agent response text
    Agent {
        content: String,
        timestamp: DateTime<Utc>,
    },
    /// Agent thinking/reasoning
    Thought {
        content: String,
        timestamp: DateTime<Utc>,
    },
    /// A tool call
    ToolCall {
        call_id: String,
        title: String,
        kind: String,
        input: Option<serde_json::Value>,
        output: Option<serde_json::Value>,
        status: String,
        timestamp: DateTime<Utc>,
    },
}

impl HistoryMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
            timestamp: Utc::now(),
        }
    }

    pub fn agent(content: impl Into<String>) -> Self {
        Self::Agent {
            content: content.into(),
            timestamp: Utc::now(),
        }
    }

    #[allow(dead_code)]
    pub fn thought(content: impl Into<String>) -> Self {
        Self::Thought {
            content: content.into(),
            timestamp: Utc::now(),
        }
    }

    pub fn tool_call(
        call_id: impl Into<String>,
        title: impl Into<String>,
        kind: impl Into<String>,
    ) -> Self {
        Self::ToolCall {
            call_id: call_id.into(),
            title: title.into(),
            kind: kind.into(),
            input: None,
            output: None,
            status: "in_progress".to_string(),
            timestamp: Utc::now(),
        }
    }
}

/// Conversation history for a single session
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MessageHistory {
    /// The session ID this history belongs to
    pub session_id: String,
    /// All messages in chronological order
    pub messages: Vec<HistoryMessage>,
}

impl MessageHistory {
    /// Create a new empty history for a session
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            messages: Vec::new(),
        }
    }

    /// Get the history directory path
    fn history_dir() -> Option<PathBuf> {
        dirs::data_local_dir().map(|p| p.join("cursor-acp").join("history"))
    }

    /// Get the file path for a specific session's history
    fn history_path(session_id: &str) -> Option<PathBuf> {
        Self::history_dir().map(|p| p.join(format!("{}.json", session_id)))
    }

    /// Load history for a session from disk
    pub fn load(session_id: &str) -> Self {
        let Some(path) = Self::history_path(session_id) else {
            tracing::warn!("Could not determine history path");
            return Self::new(session_id);
        };

        if !path.exists() {
            return Self::new(session_id);
        }

        match fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(history) => {
                    tracing::debug!("Loaded message history from {:?}", path);
                    history
                }
                Err(e) => {
                    tracing::warn!("Failed to parse message history: {}", e);
                    Self::new(session_id)
                }
            },
            Err(e) => {
                tracing::warn!("Failed to read message history: {}", e);
                Self::new(session_id)
            }
        }
    }

    /// Save history to disk
    pub fn save(&self) {
        let Some(path) = Self::history_path(&self.session_id) else {
            tracing::warn!("Could not determine history path");
            return;
        };

        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                tracing::warn!("Failed to create history directory: {}", e);
                return;
            }
        }

        match serde_json::to_string_pretty(self) {
            Ok(contents) => {
                if let Err(e) = fs::write(&path, contents) {
                    tracing::warn!("Failed to write message history: {}", e);
                } else {
                    tracing::debug!("Saved message history to {:?}", path);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to serialize message history: {}", e);
            }
        }
    }

    /// Add a message to the history and save
    pub fn push(&mut self, message: HistoryMessage) {
        self.messages.push(message);
        self.save();
    }

    /// Update the last tool call with completion info
    pub fn complete_tool_call(
        &mut self,
        call_id: &str,
        output: Option<serde_json::Value>,
        status: &str,
    ) {
        // Find the tool call and update it
        for msg in self.messages.iter_mut().rev() {
            if let HistoryMessage::ToolCall {
                call_id: id,
                output: ref mut out,
                status: ref mut st,
                ..
            } = msg
            {
                if id == call_id {
                    *out = output;
                    *st = status.to_string();
                    break;
                }
            }
        }
        self.save();
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Get the number of messages
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Delete history file from disk
    #[allow(dead_code)]
    pub fn delete(session_id: &str) {
        if let Some(path) = Self::history_path(session_id) {
            if path.exists() {
                if let Err(e) = fs::remove_file(&path) {
                    tracing::warn!("Failed to delete history file: {}", e);
                } else {
                    tracing::debug!("Deleted history file: {:?}", path);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let user_msg = HistoryMessage::user("Hello");
        assert!(matches!(user_msg, HistoryMessage::User { .. }));

        let agent_msg = HistoryMessage::agent("Hi there");
        assert!(matches!(agent_msg, HistoryMessage::Agent { .. }));

        let tool_msg = HistoryMessage::tool_call("123", "Read file", "read");
        assert!(matches!(tool_msg, HistoryMessage::ToolCall { .. }));
    }

    #[test]
    fn test_history_operations() {
        let mut history = MessageHistory::new("test-session");
        assert!(history.is_empty());

        history.messages.push(HistoryMessage::user("Test"));
        assert_eq!(history.len(), 1);
    }
}
