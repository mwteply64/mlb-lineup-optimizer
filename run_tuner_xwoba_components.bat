@echo off
cd /d "%~dp0"
python tuner_xwoba_components.py
echo Done. Results saved to tuner_xwoba_components_results_%DATE:~10,4%-%DATE:~4,2%-%DATE:~7,2%.txt
pause
