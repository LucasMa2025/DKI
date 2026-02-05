@echo off
REM DKI System Startup Script for Windows
REM ======================================

echo ========================================
echo   DKI - Dynamic KV Injection System
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Set working directory
cd /d "%~dp0.."
echo Working directory: %CD%

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual environment not found. Using system Python.
)

REM Initialize database if needed
if not exist "data" mkdir data
if not exist "data\dki.db" (
    echo Initializing database...
    sqlite3 data\dki.db < scripts\init_db.sql 2>nul
    if errorlevel 1 (
        echo [INFO] sqlite3 not found, database will be created by SQLAlchemy
    )
)

REM Create experiment results directory
if not exist "experiment_results" mkdir experiment_results

REM Set environment variables
set DKI_CONFIG_PATH=config\config.yaml
set PYTHONPATH=%CD%

REM Parse arguments
set "MODE=web"
if "%1"=="web" set "MODE=web"
if "%1"=="api" set "MODE=api"
if "%1"=="experiment" set "MODE=experiment"
if "%1"=="generate-data" set "MODE=generate-data"

echo.
echo Starting in %MODE% mode...
echo.

if "%MODE%"=="web" (
    echo Starting Web UI at http://localhost:8080
    python -m dki.web.app
)

if "%MODE%"=="api" (
    echo Starting API server at http://localhost:8080
    uvicorn dki.web.app:create_app --factory --host 0.0.0.0 --port 8080
)

if "%MODE%"=="experiment" (
    echo Running experiments...
    python -m dki.experiment.runner
)

if "%MODE%"=="generate-data" (
    echo Generating experiment data...
    python -m dki.experiment.data_generator
)

pause
