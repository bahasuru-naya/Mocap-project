@echo off
setlocal
cd /d "%~dp0"

call "my_env\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate my_env.
    exit /b 1
)

python "src\convert_model_to_onnx.py"
if errorlevel 1 (
    echo Model conversion failed.
    exit /b 1
)

endlocal
