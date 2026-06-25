@echo off
chcp 65001 > nul
cd /d %~dp0
echo venv を作成するのだ...
py -3.10 -m venv venv
echo パッケージをインストールするのだ...
venv\Scripts\pip install -r requirements.txt
echo セットアップ完了なのだ！ run.bat で起動できるのだ。
pause
