@echo off
chcp 65001 >nul 2>&1

:: BookTerm Gacha - Main Application Launcher
:: Auto-checks environment and installs dependencies on first run

cd /d "%~dp0"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo   ERROR: Python not found!
    echo ============================================================
    echo.
    echo   Please install Python 3.10+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo   Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Start application (environment check is built-in)
python.exe app.py

:: Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Program exited with error.
    pause
)