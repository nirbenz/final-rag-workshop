# RAG Workshop

A hands-on workshop for learning Retrieval-Augmented Generation (RAG) through implementing chunking and retrieval strategies. Build a production-ready RAG system in 4 hours.

## Quick Start

### Using `uv` (same instructions for all operating systems)

### MacOS

```bash
# Verify Python 3.12
python3 --version  # Should show Python 3.12.x

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
pip3 install -e

# Verify setup
python3 scripts/verify_setup.py

# Run the workshop application
python3 -m nicegui_app.main
```

### Linux/WSL

**Inside WSL - make sure you are within the home directory**

```bash
cd ~
```

```bash
# update + install python3.12
sudo apt update
sudo apt install -y python3.12-full
# Verify Python 3.12
python3 --version  # Should show Python 3.12.x

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
pip3 install -e

# Verify setup
python3 scripts/verify_setup.py

# Run the workshop application
python3 -m nicegui_app.main
```

### Windows (Non-WSL)

**TBD**

```powershell
# Verify Python 3.12
python --version  # Should show Python 3.12.x

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python scripts\verify_setup.py

# Run the workshop application
python -m nicegui_app.main
```

### (Conditional) Using `uv` (same instructions for all operating systems)

> if not using UV, skip

```bash
# install uv; use homebrew, curl or whatever works on Windows

uv sync
uv run python scripts/verify_setup.py
uv run python -m nicegui_app.main
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
