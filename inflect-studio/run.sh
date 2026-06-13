#!/usr/bin/env bash
# Inflect Studio launcher (Linux/macOS). Creates a venv, installs deps once, runs.
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null

if [ ! -f ".venv/.deps_installed" ]; then
    if ! python -c "import torch" 2>/dev/null; then
        echo "Installing PyTorch..."
        # Linux: CUDA wheels. macOS: falls back to the default (CPU/MPS) index.
        if [ "$(uname)" = "Darwin" ]; then
            python -m pip install torch torchaudio
        else
            python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
        fi
    fi
    echo "Installing requirements (this can take a while the first time)..."
    python -m pip install -r requirements.txt
    touch .venv/.deps_installed
fi

exec python -m inflect
