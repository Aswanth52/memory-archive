@echo off
title Memory Archive - Setup
echo ================================================
echo  Memory Archive - First Time Setup
echo ================================================
echo.
echo This will install all required dependencies.
echo This may take 10-15 minutes depending on your internet speed.
echo Do not close this window.
echo.
 
:: Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed on this computer.
    echo Please download and install Python 3.10 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit
)
 
echo Python found. Creating virtual environment...
python -m venv python_env
 
echo.
echo Upgrading pip...
python_env\Scripts\python.exe -m pip install --upgrade pip --quiet
 
echo.
echo Installing dependencies...
python_env\Scripts\pip install fastapi uvicorn open-clip-torch faiss-cpu deepface pillow numpy torch tf-keras
 
echo.
echo ================================================
echo  Setup complete! Run run.bat to start the app.
echo ================================================
pause