@echo off
ECHO HyperSyncDT Startup Script

REM Check if virtual environment exists and create if not
IF NOT EXIST venv (
    ECHO Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
CALL venv\Scripts\activate.bat

REM Install dependencies if not already installed
pip show streamlit >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Installing dependencies...
    pip install -r requirements.txt
)

REM Run the application
ECHO Starting HyperSyncDT application...
streamlit run frontend/app.py

PAUSE 