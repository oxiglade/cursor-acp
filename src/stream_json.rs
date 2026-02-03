//! Types for parsing Cursor CLI's stream-json output format.
//!
//! When running `cursor -p --output-format stream-json`, Cursor emits
//! newline-delimited JSON events that we parse into these types.

use serde::{Deserialize, Serialize};

/// A stream event from Cursor CLI
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// System initialization event (emitted once per session)
    System(SystemEvent),
    /// User message event
    User(MessageEvent),
    /// Assistant message event
    Assistant(MessageEvent),
    /// Thinking event (for models with extended thinking)
    Thinking(ThinkingEvent),
    /// Tool call event
    ToolCall(ToolCallEvent),
    /// Final result event
    Result(ResultEvent),
}

/// System initialization event
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemEvent {
    pub subtype: String,
    #[serde(default)]
    pub api_key_source: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub permission_mode: Option<String>,
}

/// Message event (user or assistant)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageEvent {
    pub message: Message,
}

/// Thinking event (for models with extended thinking)
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingEvent {
    pub subtype: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub timestamp_ms: Option<u64>,
}

/// A chat message
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: Vec<ContentPart>,
}

/// Content part within a message
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image { url: String },
}

/// Tool call event
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallEvent {
    pub subtype: ToolCallSubtype,
    pub call_id: String,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub arguments: Option<serde_json::Value>,
    #[serde(default)]
    pub result: Option<ToolCallResult>,
}

/// Tool call subtype
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallSubtype {
    Started,
    Completed,
}

/// Result of a tool call
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallResult {
    #[serde(default)]
    pub success: bool,
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    /// For file operations
    #[serde(default)]
    pub lines_read: Option<u64>,
    #[serde(default)]
    pub lines_written: Option<u64>,
    #[serde(default)]
    pub file_size: Option<u64>,
}

/// Final result event
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResultEvent {
    pub subtype: String,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub result: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub duration_api_ms: Option<u64>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub request_id: Option<String>,
}

impl StreamEvent {
    /// Parse a stream-json line into an event
    pub fn parse(line: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_system_event() {
        let json = r#"{"type":"system","subtype":"init","model":"gpt-4","cwd":"/home/user"}"#;
        let event = StreamEvent::parse(json).unwrap();
        match event {
            StreamEvent::System(sys) => {
                assert_eq!(sys.subtype, "init");
                assert_eq!(sys.model, Some("gpt-4".to_string()));
            }
            _ => panic!("expected system event"),
        }
    }

    #[test]
    fn test_parse_tool_call_started() {
        let json = r#"{"type":"tool_call","subtype":"started","callId":"abc123","toolName":"readFile","arguments":{"path":"test.rs"}}"#;
        let event = StreamEvent::parse(json).unwrap();
        match event {
            StreamEvent::ToolCall(tc) => {
                assert_eq!(tc.subtype, ToolCallSubtype::Started);
                assert_eq!(tc.call_id, "abc123");
                assert_eq!(tc.tool_name, Some("readFile".to_string()));
            }
            _ => panic!("expected tool call event"),
        }
    }

    #[test]
    fn test_parse_result() {
        let json = r#"{"type":"result","subtype":"success","is_error":false,"result":"Done!","duration_ms":1234}"#;
        let event = StreamEvent::parse(json).unwrap();
        match event {
            StreamEvent::Result(res) => {
                assert_eq!(res.subtype, "success");
                assert!(!res.is_error);
                assert_eq!(res.result, Some("Done!".to_string()));
            }
            _ => panic!("expected result event"),
        }
    }
}
