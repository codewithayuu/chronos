@echo off
echo CHRONOS is starting up!
echo.

start "CHRONOS Backend" echo Starting Backend...
cd project-chronos
start cmd /k ".venv\Scripts\python run_server.py --port 8000 --speed 3"

timeout /t 3 /nobreak >nul

start "CHRONOS Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Both services are starting in separate windows...
