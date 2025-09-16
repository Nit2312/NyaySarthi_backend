@echo off
cd /d %~dp0
call conda activate .\venv
uvicorn main:app --reload
pause
