@echo off
cd /d "%~dp0"
python build_age_data.py
if %errorlevel% neq 0 (
    echo.
    echo *** SCRIPT FAILED - see error output above ***
) else (
    echo Done. Results saved to player_ages.csv
)
pause
