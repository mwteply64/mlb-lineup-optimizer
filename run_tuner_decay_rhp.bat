@echo off
cd /d "%~dp0"
python tuner_decay_rhp.py
echo Done. Results saved to tuner_decay_rhp_results_%DATE:~10,4%-%DATE:~4,2%-%DATE:~7,2%.txt
pause
