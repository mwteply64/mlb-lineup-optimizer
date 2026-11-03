@echo off
cd /d "%~dp0"
python update_live_data.py > run_update_live_data_log.txt 2>&1
echo Done. Output saved to run_update_live_data_log.txt
pause
