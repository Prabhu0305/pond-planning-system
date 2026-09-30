#!/bin/bash
cd "$(dirname "$0")"

echo "=== Process Status ==="
ps aux | grep -E 'python.*app.py' | grep -v grep || echo "Server is NOT running!"

echo ""
echo "=== Health Probe ==="
curl -s http://127.0.0.1:3000/health || echo "Connection failed!"

echo ""
echo "=== Latest Server Logs ==="
tail -n 12 flask.log 2>/dev/null || echo "No log file found."
