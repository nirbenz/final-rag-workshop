#!/usr/bin/env python3
"""
RAG Workshop Setup Verification Script

Run this script before the workshop to verify your environment is ready.

Usage (any of these should work):
    python scripts/verify_setup.py
    uv run python scripts/verify_setup.py
    poetry run python scripts/verify_setup.py
"""

import os
from pathlib import Path
import sys


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 50}")
    print(f"  {text}")
    print("=" * 50)


def print_result(name: str, passed: bool, message: str) -> None:
    """Print a check result."""
    status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"\n[{status}] {name}")
    for line in message.split("\n"):
        print(f"       {line}")


def check_python_version() -> tuple[bool, str]:
    """Check Python version is 3.12+."""
    version = sys.version_info
    if version < (3, 12):
        return False, (
            f"Python 3.12+ required, found {version.major}.{version.minor}\n"
            "Install Python 3.12+ from https://python.org"
        )
    return True, f"Python {version.major}.{version.minor}.{version.micro}"


def check_virtual_environment() -> tuple[bool, str]:
    """Check we're running inside a virtual environment."""
    # Check common venv indicators
    in_venv = (
        hasattr(sys, "real_prefix")  # virtualenv
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)  # venv
        or os.environ.get("VIRTUAL_ENV")  # explicit env var
        or os.environ.get("CONDA_DEFAULT_ENV")  # conda
        or os.environ.get("POETRY_ACTIVE")  # poetry
    )

    if not in_venv:
        return False, (
            "Not running inside a virtual environment!\n"
            "Activate your venv first:\n"
            "  source .venv/bin/activate  (Linux/Mac)\n"
            "  .venv\\Scripts\\activate   (Windows)\n"
            "Or use: uv run python scripts/verify_setup.py"
        )

    venv_path = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_DEFAULT_ENV") or sys.prefix
    return True, f"Virtual environment: {venv_path}"


def check_core_imports() -> tuple[bool, str]:
    """Check critical workshop dependencies can be imported."""
    required = {
        "numpy": "NumPy for vector operations",
        "nicegui": "NiceGUI for the workshop UI",
        "pydantic": "Pydantic for data models",
        "pydantic_ai": "Pydantic-AI for LLM integration",
        "qdrant_client": "Qdrant for vector database",
        "tiktoken": "Tiktoken for token counting",
        "whatstk": "Whatstk for WhatsApp parsing",
    }

    passed = []
    failed = []

    for module, description in required.items():
        try:
            __import__(module)
            passed.append(f"{module}")
        except ImportError as e:
            failed.append(f"{module} ({description}): {e}")

    if failed:
        msg = "Missing dependencies:\n" + "\n".join(f"  - {f}" for f in failed)
        msg += "\n\nRun: uv sync  OR  pip install -r requirements.txt"
        return False, msg

    return True, f"All {len(passed)} core packages imported successfully"


def check_optional_imports() -> tuple[bool, str]:
    """Check optional but recommended dependencies."""
    optional = {
        "dotenv": "python-dotenv for .env file loading",
        "loguru": "Loguru for logging",
        "httpx": "HTTPX for async HTTP",
    }

    warnings = []
    for module, description in optional.items():
        try:
            __import__(module)
        except ImportError:
            warnings.append(f"{module} ({description})")

    if warnings:
        return True, "Optional packages missing (non-critical):\n" + "\n".join(f"  - {w}" for w in warnings)

    return True, "All optional packages available"


def check_api_keys() -> tuple[bool, str]:
    """Check at least one LLM API key is configured."""
    # Try to load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv

        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    keys_to_check = {  # pragma: allowlist secret
        "OPENAI_API_KEY": "OpenAI",  # pragma: allowlist secret
        "ANTHROPIC_API_KEY": "Anthropic",  # pragma: allowlist secret
        "GOOGLE_API_KEY": "Google AI",  # pragma: allowlist secret
        "GOOGLE_CLOUD_PROJECT": "Google Vertex AI",
    }

    found = []
    for key, provider in keys_to_check.items():
        value = os.environ.get(key)
        if value and len(value) > 5:  # Basic sanity check
            # Mask the key for display
            masked = value[:4] + "..." + value[-4:] if len(value) > 10 else "***"
            found.append(f"{provider} ({key}={masked})")

    if not found:
        return False, (
            "No API keys found!\n"
            "Create a .env file with at least one of:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "  ANTHROPIC_API_KEY=sk-ant-...\n"
            "  GOOGLE_API_KEY=..."
        )

    return True, "API keys configured:\n" + "\n".join(f"  - {f}" for f in found)


def check_chat_files() -> tuple[bool, str]:
    """Check sample chat files exist."""
    chats_dir = Path("chats")

    if not chats_dir.exists():
        return False, (
            "chats/ directory not found!\n"
            "Create it and add a WhatsApp export:\n"
            "  mkdir chats\n"
            "  cp ~/Downloads/WhatsApp*.txt chats/"
        )

    txt_files = list(chats_dir.glob("*.txt"))
    zip_files = list(chats_dir.glob("*.zip"))

    if zip_files and not txt_files:
        return False, (
            f"Found {len(zip_files)} .zip file(s) but no .txt files.\n"
            "Unzip the WhatsApp export first:\n"
            f"  unzip {zip_files[0]} -d chats/"
        )

    if not txt_files:
        return False, (
            "No WhatsApp chat files found in chats/\n"
            "Export a chat from WhatsApp (without media) and place the .txt file here."
        )

    # Check file sizes
    files_info = []
    for f in txt_files[:5]:  # Show first 5
        size_kb = f.stat().st_size / 1024
        files_info.append(f"{f.name} ({size_kb:.1f} KB)")

    return True, f"Found {len(txt_files)} chat file(s):\n" + "\n".join(f"  - {f}" for f in files_info)


def check_workshop_imports() -> tuple[bool, str]:
    """Check workshop-specific modules can be imported."""
    import importlib.util

    modules_to_check = [
        "workshop.chat",
        "workshop.rag.chunkers",
        "workshop.rag.engines",
    ]

    for module in modules_to_check:
        spec = importlib.util.find_spec(module)
        if spec is None:
            return False, (f"Workshop module '{module}' not found.\n" "Make sure you're in the project root directory.")

    return True, "Workshop modules import successfully"


def check_working_directory() -> tuple[bool, str]:
    """Check we're in the correct working directory."""
    cwd = Path.cwd()

    # Look for project markers
    markers = ["pyproject.toml", "src/workshop", "src/nicegui_app"]
    found = [m for m in markers if (cwd / m).exists()]

    if len(found) < 2:
        return False, (
            f"Current directory: {cwd}\n"
            "This doesn't look like the workshop root.\n"
            "cd to the directory containing pyproject.toml"
        )

    return True, f"Working directory: {cwd}"


def main() -> int:
    """Run all verification checks."""
    print_header("RAG Workshop Setup Verification")

    checks = [
        ("Working Directory", check_working_directory),
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Core Dependencies", check_core_imports),
        ("Optional Dependencies", check_optional_imports),
        ("API Keys", check_api_keys),
        ("Chat Files", check_chat_files),
        ("Workshop Modules", check_workshop_imports),
    ]

    results = []
    for name, check_fn in checks:
        try:
            passed, message = check_fn()
        except Exception as e:
            passed, message = False, f"Check failed with error: {e}"

        print_result(name, passed, message)
        results.append(passed)

    # Summary
    print_header("Summary")

    passed_count = sum(results)
    total_count = len(results)

    if all(results):
        print("\n\033[92m" + "All checks passed! You're ready for the workshop." + "\033[0m")
        print("\nTo start the workshop UI:")
        print("  python -m nicegui_app.main")
        print("  # OR: uv run python -m nicegui_app.main")
        return 0
    else:
        failed_count = total_count - passed_count
        print(f"\n\033[91m{failed_count} check(s) failed.\033[0m")
        print("Please fix the issues above before the workshop.")
        print("\nNeed help? Contact: <instructor-email>")
        return 1


if __name__ == "__main__":
    sys.exit(main())
