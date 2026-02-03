# cursor-acp

An [Agent Client Protocol (ACP)](https://agentclientprotocol.com/) adapter for [Cursor CLI](https://cursor.com/cli), enabling Cursor to work with ACP-compatible development environments like [Zed](https://zed.dev/).

## Overview

cursor-acp bridges the Cursor CLI with the Agent Client Protocol, allowing you to use Cursor's AI coding capabilities within Zed and other ACP-compatible editors.

## Features

- Browser-based login flow (via `cursor-agent login`)
- Prompt processing via Cursor CLI
- Streaming responses with real-time updates
- Tool call notifications (file read/write, command execution)
- Session management
- Model selection

## Installation

### From Source

```bash
git clone https://github.com/oxideai/cursor-acp
cd cursor-acp
cargo install --path .
```

### Prerequisites

- [Rust](https://rustup.rs/) toolchain
- [Cursor CLI](https://cursor.com/cli) installed and in your PATH
  ```bash
  curl https://cursor.com/install -fsSL | bash
  ```

## Authentication

cursor-acp supports two authentication methods:

### 1. Browser Login (Recommended)

When you first use cursor-acp, it will prompt you to authenticate via browser. If the client (like Zed) supports it, it will automatically launch the login flow. Otherwise, you can run:

```bash
cursor-agent login
```

This opens a browser to authenticate with your Cursor account.

### 2. API Key

Alternatively, set the `CURSOR_API_KEY` environment variable:

```bash
export CURSOR_API_KEY=your_api_key_here
```

## Usage

### Running

```bash
cursor-acp
```

The adapter communicates over stdio using the ACP JSON-RPC protocol.

### With Zed

Configure Zed to use cursor-acp as an AI agent (documentation coming soon).

## Architecture

```
┌─────────┐     ACP (JSON-RPC/stdio)     ┌─────────────┐     spawn + stream-json     ┌─────────────┐
│   Zed   │ ◄──────────────────────────► │  cursor-acp │ ◄─────────────────────────► │ Cursor CLI  │
└─────────┘                              └─────────────┘                              └─────────────┘
```

cursor-acp:
1. Receives ACP protocol messages from the editor (Zed)
2. Translates prompts to Cursor CLI commands
3. Spawns Cursor CLI in headless mode with `--output-format stream-json`
4. Parses streaming JSON events from Cursor
5. Sends ACP session updates back to the editor

## Development

```bash
# Build
cargo build

# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run
```

## License

Apache-2.0

## Related Projects

- [codex-acp](https://github.com/zed-industries/codex-acp) - ACP adapter for OpenAI Codex
- [claude-code-acp](https://github.com/zed-industries/claude-code-acp) - ACP adapter for Claude Code
- [Agent Client Protocol](https://agentclientprotocol.com/) - The protocol specification
