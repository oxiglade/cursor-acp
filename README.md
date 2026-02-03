# ACP adapter for Cursor

Use [Cursor](https://cursor.com/) from [ACP-compatible](https://agentclientprotocol.com) clients such as [Zed](https://zed.dev)!

This tool implements an ACP adapter around the Cursor CLI, supporting:

- Context @-mentions
- Images
- Tool calls (with permission requests)
- Extended thinking
- Session history persistence
- Model selection (Claude, GPT, Gemini)
- Mode selection (default, plan, ask)
- Auth Methods:
  - Browser login (via `cursor-agent login`)
  - CURSOR_API_KEY

Learn more about the [Agent Client Protocol](https://agentclientprotocol.com/).

## How to use

### Zed

Once registered in the ACP registry, Zed will be able to use this adapter out of the box.

To use Cursor, open the Agent Panel and click "New Cursor Thread" from the `+` button menu in the top-right.

Read the docs on [External Agent](https://zed.dev/docs/ai/external-agents) support.

### Other clients

Or try it with any of the other [ACP compatible clients](https://agentclientprotocol.com/overview/clients)!

#### Prerequisites

[Cursor CLI](https://cursor.com/cli) must be installed:

```bash
curl https://cursor.com/install -fsSL | bash
```

#### Installation

Install the adapter from the latest release for your architecture and OS: https://github.com/oxideai/cursor-acp/releases

You can then use `cursor-acp` as a regular ACP agent:

```
cursor-acp
```

## License

Apache-2.0
