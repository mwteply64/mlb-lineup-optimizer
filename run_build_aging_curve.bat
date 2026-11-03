@echo off
cd /d "%~dp0"
python build_aging_curve.py
echo Done. Results saved to aging_curve_results_%DATE:~10,4%-%DATE:~4,2%-%DATE:~7,2%.txt
pause
