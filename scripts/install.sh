#!/bin/bash

set -e

REQUIRED_PYTHON="3.12"

if [ "$(uname)" == "Linux" ]; then
    echo "Detected Linux/WSL"
    sudo apt update
    sudo apt install -y python3.12-full
elif [ "$(uname)" == "Darwin" ]; then
    echo "Detected MacOS"
    brew install python@3.12
else
    echo "Unsupported OS: $(uname)"
    echo "For Windows (non-WSL), please follow the manual instructions in README.md"
    exit 1
fi

hash -r

PYTHON_VERSION=$(python3.12 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f1,2)

if [ "$PYTHON_MAJOR_MINOR" != "$REQUIRED_PYTHON" ]; then
    echo "Error: Python $REQUIRED_PYTHON is required, but found Python $PYTHON_VERSION"
    exit 1
fi

echo "Python $PYTHON_VERSION found"
echo "Updating pip..."

echo "Creating virtual environment..."
python3.12 -m pip install -U pip
python3.12 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip3 install -r requirements.txt
pip3 install -e .

echo "Copying example.env to .env..."
# if .env does not exist
if [ ! -f .env ]; then
    cp example.env .env
    echo "Example .env file created. Please fill in your API keys."
else
    echo ".env file already exists. Skipping copy."
fi

echo "Verifying setup..."
python3.12 scripts/verify_setup.py

echo ""
echo "Setup complete! To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the workshop application, run:"
echo "  python3 -m nicegui_app.main"
