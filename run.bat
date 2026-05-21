
@echo off
title Memory Archive
if not exist "%~dp0python_env\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Please run setup.bat first.
    echo.
    pause
    exit
)
echo Starting Memory Archive...
start "" "%~dp0python_env\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
echo Waiting for server to start...
timeout /t 8 /nobreak > nul
start http://127.0.0.1:8000