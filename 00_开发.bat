@echo off
chcp 65001 > nul

cd /d "%~dp0"

set "PATH=%~dp0resource;%PATH%"

echo Starting BookTerm Gacha - Experiment GUI...
python "frontend\new_gui\app.py"

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
    pause
)
