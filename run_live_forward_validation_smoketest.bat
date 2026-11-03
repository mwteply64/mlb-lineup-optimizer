@echo off
cd /d "%~dp0"
python live_forward_validation.py --smoke-test > run_live_forward_validation_smoketest_log.txt 2>&1
echo Done. Output saved to run_live_forward_validation_smoketest_log.txt
echo.
echo Open that file and check:
echo   1. No ERROR lines (especially missing-column or import errors)
echo   2. real_avg / opt_avg are plausible wOBA values (roughly 0.200-0.450)
echo   3. n_real / n_opt are both at least 5
echo Only run run_live_forward_validation.bat (the full pull) after this looks right.
pause
