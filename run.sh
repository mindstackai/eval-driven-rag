#!/usr/bin/env bash
set -e

source .venv/bin/activate
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. streamlit run src/app.py
