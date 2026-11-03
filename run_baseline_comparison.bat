@echo off
cd /d "%~dp0"
python baseline_comparison.py
echo Done. Results saved to baseline_comparison_results_%DATE:~10,4%-%DATE:~4,2%-%DATE:~7,2%.txt
pause
