@echo off
cd /d %~dp0
call conda activate .\venv
uvicorn app:app --reload --host 0.0.0.0 --port 8000
pause
