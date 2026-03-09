#!/usr/bin/env bash
set -e

echo "Setting up eval-driven-rag..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created virtual environment at .venv"
fi

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Copy .env.example if .env doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example — add your OPENAI_API_KEY before running."
fi

echo ""
echo "Setup complete."
echo "  Activate env : source .venv/bin/activate"
echo "  Ingest docs  : python -m src.ingest"
echo "  Run Streamlit: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. streamlit run src/app.py"
echo "  Run eval     : python -m src.eval.run_eval --qa_pairs data/eval/qa_pairs.json --k 5"
