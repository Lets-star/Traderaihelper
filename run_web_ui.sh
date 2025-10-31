#!/usr/bin/env bash

set -e

if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed."
    echo "Please install project dependencies first:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "Starting Token Charts & Indicators Dashboard..."
echo "Open http://localhost:8501 in your browser."

streamlit run web_ui.py
