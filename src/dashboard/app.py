"""Standalone entry point for the eval dashboard.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. streamlit run src/dashboard/app.py
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

from src.dashboard import render_dashboard

render_dashboard()
