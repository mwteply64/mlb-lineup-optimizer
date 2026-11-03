@echo off
cd /d "%~dp0"
python download_team_tags.py --smoke-test > run_team_tags_smoketest_log.txt 2>&1
echo Done. Output saved to run_team_tags_smoketest_log.txt
echo.
echo Open that file and check:
echo   1. "Columns OK" line appears (not a missing-columns error)
echo   2. The sample team codes look like real MLB teams (NYY, LAD, etc.)
echo   3. Nobody's split across two teams unless they were actually traded
echo Only run run_team_tags.bat (the full pull) after this looks right.
pause
