#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
echo "Starting TrendLine... your browser will open automatically."
echo "To stop: close this Terminal window, or press Control+C."
streamlit run app.py
