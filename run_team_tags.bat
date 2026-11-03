@echo off
cd /d "%~dp0"
python download_team_tags.py > run_team_tags_log.txt 2>&1
echo Done. Output saved to run_team_tags_log.txt
pause
