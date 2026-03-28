#!/bin/sh
PORT="${PORT:-8501}"
exec streamlit run app/dashboard.py --server.port="$PORT" --server.address=0.0.0.0 --server.headless=true
