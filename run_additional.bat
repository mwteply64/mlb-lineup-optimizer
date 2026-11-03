@echo off
cd /d "%~dp0"
python download_additional_data.py > run_additional_log.txt 2>&1
echo Done. Output saved to run_additional_log.txt
pause
