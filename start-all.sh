#!/bin/bash
echo "CHRONOS is starting up!"

# Start Backend
cd project-chronos
.venv/bin/python3 run_server.py --port 8000 --speed 3 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start Frontend
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "Backend running on PID $BACKEND_PID"
echo "Frontend running on PID $FRONTEND_PID"
echo "To stop, press Ctrl+C"

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
