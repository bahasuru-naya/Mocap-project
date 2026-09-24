@echo off
setlocal
cd /d "%~dp0"

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate the virtual environment.
    pause
    exit /b 1
)

python "src\app_main.py"
if errorlevel 1 (
    echo.
    echo The application exited with an error.
    pause
)

endlocal
