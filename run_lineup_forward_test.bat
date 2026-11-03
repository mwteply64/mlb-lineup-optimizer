@echo off
cd /d "%~dp0"
python lineup_forward_test.py > run_lineup_forward_test_log.txt 2>&1
echo Done. Output saved to run_lineup_forward_test_log.txt
echo (Full results also saved to lineup_forward_test_results_<date>.txt)
pause
