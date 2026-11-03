@echo off
cd /d "%~dp0"
python download_v3_expansion.py > run_v3_expansion_log.txt 2>&1
echo Done. Output saved to run_v3_expansion_log.txt
pause
