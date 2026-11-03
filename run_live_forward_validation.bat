@echo off
cd /d "%~dp0"
python live_forward_validation.py > run_live_forward_validation_log.txt 2>&1
echo Done. Output saved to run_live_forward_validation_log.txt
echo (Full results also saved to live_forward_validation_results_<date>.txt)
echo (Summary saved to live_forward_validation_summary.json -- test.py reads this.)
pause
