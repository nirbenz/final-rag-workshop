# RAG Workshop

A hands-on workshop for learning Retrieval-Augmented Generation (RAG) through implementing chunking and retrieval strategies. Build a production-ready RAG system in 4 hours.

## Quick Start

### MacOS

```bash
# Install uv (package manager)
brew install uv

# Install dependencies
uv sync

# Run the workshop application
uv run python -m nicegui_app.main

# Alternative: For environments with self-signed certificates (e.g., corporate proxies)
uv run python run_app.py
```

### Linux/WSL

```bash
# Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart shell or source the env
source $HOME/.local/bin/env

# Install dependencies
uv sync

# Run the workshop application
uv run python -m nicegui_app.main

# Alternative: For environments with self-signed certificates
uv run python run_app.py
```

### Windows (Non-WSL)

See [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2).

```powershell
# Install uv (package manager) using PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart terminal, then install dependencies
uv sync

# Run the workshop application
uv run python -m nicegui_app.main

# Alternative: For environments with self-signed certificates
uv run python run_app.py
```

## Environment Variables

Set these in a `.env` file or export them:

```bash
export GOOGLE_API_KEY="your-key-here"           # For Google AI Studio
export GOOGLE_CLOUD_PROJECT="your-project"      # For Vertex AI
export GOOGLE_CLOUD_LOCATION="us-central1"      # For Vertex AI
export OPENAI_API_KEY="your-key-here"           # For OpenAI
export ANTHROPIC_API_KEY="your-key-here"        # For Anthropic

# LiteLLM Proxy Configuration (if using a proxy)
export OPENAI_API_BASE="https://your-litellm-proxy:port"  # LiteLLM proxy endpoint
export SSL_VERIFY="false"                        # Set to "false" to disable SSL verification
```

## LiteLLM Support

The workshop now supports [LiteLLM](https://docs.litellm.ai/) as a proxy for accessing multiple LLM providers through a unified OpenAI-compatible API. This is useful for:
- Corporate environments with centralized LLM access
- Cost tracking and rate limiting across teams
- Environments requiring SSL inspection bypass

### Configuration

Models are configured via Hydra configs in `configs/models/`. To use LiteLLM:
1. Set `OPENAI_API_BASE` to your LiteLLM proxy endpoint
2. Configure models using the `litellm:` prefix (e.g., `litellm:global.anthropic.claude-sonnet-4-5-20250929-v1:0`)
3. If using self-signed certificates, set `SSL_VERIFY="false"` environment variable

## Exporting WhatsApp Chat

Follow instructions [for your device](https://whatstk.readthedocs.io/en/latest/source/getting_started/export_chat.html).
Place the exported text file at `chats/default_chat.txt` or use the included `chats/example_chat.txt`.

---

## Workshop Guide

See the **[Participant Guide](docs/PARTICIPANT_GUIDE.md)** for the full hands-on walkthrough with step-by-step instructions, exercises, and troubleshooting.

---

## License

MIT
