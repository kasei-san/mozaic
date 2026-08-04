@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo Creating venv...
py -3.10 -m venv venv
if errorlevel 1 goto :nopython

echo Installing packages...
venv\Scripts\pip install -r requirements.txt
if errorlevel 1 goto :nopip

echo Setup complete! Run run.bat to start.
pause
exit /b 0

:nopython
echo.
echo [ERROR] Failed to create venv.
echo         Python 3.10 is required. Check with: py -3.10 --version
echo         Get it from https://www.python.org/downloads/
pause
exit /b 1

:nopip
echo.
echo [ERROR] Failed to install packages from requirements.txt.
echo         Check your network connection and run setup.bat again.
pause
exit /b 1
