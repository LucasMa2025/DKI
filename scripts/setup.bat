@echo off
REM DKI System Setup Script for Windows
REM ====================================

echo ========================================
echo   DKI System Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Set working directory
cd /d "%~dp0.."
echo Working directory: %CD%

REM Create virtual environment
echo.
echo [1/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo [3/5] Installing dependencies...
pip install -r requirements.txt

REM Create directories
echo.
echo [4/5] Creating directories...
if not exist "data" mkdir data
if not exist "experiment_results" mkdir experiment_results
if not exist "logs" mkdir logs

REM Initialize database
echo.
echo [5/5] Initializing database...
python -c "from dki.database.connection import DatabaseManager; DatabaseManager()"
echo Database initialized.

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To start the system:
echo   scripts\start.bat web       - Start Web UI
echo   scripts\start.bat api       - Start API server
echo   scripts\start.bat experiment - Run experiments
echo.
pause
