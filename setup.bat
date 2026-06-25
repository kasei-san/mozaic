@echo off
chcp 65001 > nul
cd /d %~dp0
echo Creating venv...
py -3.10 -m venv venv
echo Installing packages...
venv\Scripts\pip install -r requirements.txt
echo Setup complete! Run run.bat to start.
pause
