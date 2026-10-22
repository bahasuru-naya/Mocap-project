@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Keras ensemble + scalers  -^>  ONNX
echo ============================================================

rem ---- activate existing virtual environment ----
if not exist "my_env\Scripts\activate.bat" (
    echo [ERROR] Virtual environment "my_env" not found next to this .bat file.
    goto :fail
)
call "my_env\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate Python virtual environment "my_env".
    goto :fail
)

rem ---- check input files ----
if not exist "data\" (
    echo [ERROR] "data" folder not found next to this .bat file.
    goto :fail
)
if not exist "model\" mkdir "model"

rem ---- run the converter with the venv's python ----
python convert_to_onnx.py --data-dir "data" --model-dir "model" %*
if errorlevel 1 goto :fail

echo.
echo Finished. Output: model\best_model_overall_with_scalers.onnx
goto :end

:fail
echo.
echo [ERROR] Something failed - see the messages above.

:end
echo.
pause
endlocal
