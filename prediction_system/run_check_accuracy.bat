@echo off
setlocal
cd /d "%~dp0"

echo ======================================================================
echo          Mocap Prediction System - Model Accuracy Evaluation
echo ======================================================================
echo.

call "my_env\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate Python virtual environment 'my_env'.
    pause
    exit /b 1
)

python "Test\predict_pose_from_recording.py" %*
if errorlevel 1 (
    echo.
    echo [ERROR] Evaluation encountered errors.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo Accuracy check completed successfully.
echo ======================================================================
pause
endlocal
