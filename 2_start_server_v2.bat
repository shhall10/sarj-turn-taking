@echo off
cd /d "%~dp0"
title TurnSense v2 server - keep this window open
echo Starting TurnSense v2 on port 8765 for the organisers' network.
echo IMPORTANT: close any other TurnSense server window first (only one can use port 8765).
echo If Windows Firewall asks, click "Allow access".
echo.
".venv\Scripts\python.exe" my_system.py --host 0.0.0.0 --port 8765
echo.
echo The server stopped. Press a key to close.
pause
