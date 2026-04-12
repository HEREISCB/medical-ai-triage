#!/bin/bash
# Start the LiveKit agent in the background
python run.py start &

# Start the web UI in the foreground
uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
