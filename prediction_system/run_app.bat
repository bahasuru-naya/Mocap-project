@echo off
setlocal
cd /d "%~dp0"
call "my_env\Scripts\activate.bat"
python "src\app_main.py"
endlocal
