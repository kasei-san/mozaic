@echo off
cd /d %~dp0
echo input と output のファイルを削除するのだ...
del /Q input\*
del /Q output\*
echo 完了なのだ！
pause
