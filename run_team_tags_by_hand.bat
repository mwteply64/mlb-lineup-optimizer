@echo off
cd /d "%~dp0"
python download_team_tags_by_hand.py > run_team_tags_by_hand_log.txt 2>&1
