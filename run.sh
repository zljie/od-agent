#!/bin/bash

# Find and kill any process using port 8000
PID=$(lsof -ti :8000)
if [ -n "$PID" ]; then
    echo "Killing existing process on port 8000 (PID: $PID)..."
    kill -9 $PID
    sleep 1
fi

cd "$(dirname "$0")"
echo "Starting CustomerServiceAgent on http://0.0.0.0:8000 ..."
python3 -m src.app
