#!/bin/bash
# ── SentenceStudio local launcher ──────────────────────────────────────────

cd "$(dirname "$0")"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Setting up virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
  pip install -q -r requirements.txt
else
  source venv/bin/activate
fi

# Load .env if present
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo ""
echo "  ✦ SentenceStudio running at → http://localhost:5001"
echo "  Press Ctrl+C to stop."
echo ""

open "http://localhost:5001"
FLASK_ENV=development python3 app.py
