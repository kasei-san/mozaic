@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo input と output のファイルを削除するのだ...
for /F "delims=" %%f in ('dir /b /a-d input\* 2^>nul') do if /I not "%%f"==".gitkeep" del /Q "input\%%f"
for /F "delims=" %%f in ('dir /b /a-d output\* 2^>nul') do if /I not "%%f"==".gitkeep" del /Q "output\%%f"
echo 完了なのだ！
pause
