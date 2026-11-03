@echo off
cd /d "%~dp0"
python download_woba_splits.py > run_log.txt 2>&1
echo Done. Output saved to run_log.txt
pause
