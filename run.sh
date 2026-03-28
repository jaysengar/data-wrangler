#!/bin/bash
# ─── DataWrangler — Easy Run Script ───────────────────────────────────────────
# Usage:
#   ./run.sh inference     → Run the inference script
#   ./run.sh server        → Start the FastAPI server
#   ./run.sh install       → Install/update dependencies in venv

set -e
cd "$(dirname "$0")"

VENV_DIR="./venv"
PYTHON="$VENV_DIR/bin/python3"
PIP="$VENV_DIR/bin/pip"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Install deps if needed
if ! "$PYTHON" -c "import openai" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    "$PIP" install -q -r requirements.txt
fi

case "${1:-inference}" in
    inference)
        echo "🚀 Running inference.py..."
        "$PYTHON" inference.py
        ;;
    server)
        echo "🌐 Starting FastAPI server on http://localhost:7860 ..."
        if [ -f .env ]; then
            "$VENV_DIR/bin/uvicorn" app:app --host 0.0.0.0 --port 7860 --env-file .env --reload
        else
            "$VENV_DIR/bin/uvicorn" app:app --host 0.0.0.0 --port 7860 --reload
        fi
        ;;
    install)
        echo "📦 Installing/updating all dependencies..."
        "$PIP" install -r requirements.txt
        echo "✅ Done!"
        ;;
    *)
        echo "Usage: ./run.sh [inference|server|install]"
        exit 1
        ;;
esac
