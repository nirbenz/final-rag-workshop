#! /bin/bash

# If Linux, install uv
if [ "$(uname)" == "Linux" ]; then
    sudo apt update
    curl -fsSL https://astral.sh/uv/install.sh | sh
# Else if MacOS, install uv
elif [ "$(uname)" == "Darwin" ]; then
    curl -fsSL https://astral.sh/uv/install.sh | sh
# Windows - first ensure we are in WSL
else if [ "$(uname)" == "Windows" ]; then
    if ! grep -q "WSL" /proc/version; then
        echo "Not in WSL"
        exit 1
    fi
    curl -fsSL https://astral.sh/uv/install.sh | sh
    sudo apt update
else
  echo "Unsupported OS"
  exit 1
fi

# Install dependencies
uv sync

# Run this outside of the script to activate the virtual environment
source ./.venv/bin/activate