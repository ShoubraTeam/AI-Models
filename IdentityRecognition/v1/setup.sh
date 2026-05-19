#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# --- Create venv ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with Python 3.10..."
    python3.10 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists, skipping creation."
fi

# --- Activate venv ---
source "$VENV_DIR/bin/activate"

# --- Install dependencies ---
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# --- Run server ---
echo "Starting FastAPI server..."
cd "$SCRIPT_DIR/inference"
uvicorn app:app --host 0.0.0.0 --port 3001
