@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo Deleting all files in input and output (.gitkeep is kept)...
for /F "delims=" %%f in ('dir /b /a-d input\* 2^>nul') do if /I not "%%f"==".gitkeep" del /Q "input\%%f"
for /F "delims=" %%f in ('dir /b /a-d output\* 2^>nul') do if /I not "%%f"==".gitkeep" del /Q "output\%%f"
echo Done.
pause
