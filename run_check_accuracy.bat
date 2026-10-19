@echo off
setlocal
cd /d "%~dp0prediction_system"
call run_check_accuracy.bat %*
endlocal
