#!/bin/bash
# Stop any old running instance
pkill -f 'app.py' 2>/dev/null

cd "$(dirname "$0")"

# Start using the cluster Anaconda environment (No venv needed)
nohup /opt/conda/bin/python3.14 app.py > flask.log 2>&1 < /dev/null &

echo "Starting Pond Planning System on port 3000..."
sleep 2

# Health check
echo -n "Health Check: "
curl -s http://127.0.0.1:3000/health
echo ""

# Warm up terrain cache so instant analysis is ready for grading
echo -n "Warming cache: "
curl -s http://127.0.0.1:3000/loadSample > /dev/null
curl -s http://127.0.0.1:3000/health
echo ""

MY_IP=$(hostname -I | awk '{print $1}')
echo "=============================================================="
echo "  Pond Planning System is ACTIVE in background!"
echo "  Internal Cluster URL : http://${MY_IP}:3000/planner"
echo "  Hostname URL         : http://stu55_sys2:3000/planner"
echo "  Localhost URL        : http://127.0.0.1:3000/planner"
echo "=============================================================="
