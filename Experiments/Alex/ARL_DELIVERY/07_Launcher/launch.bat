@echo off
REM ============================================================
REM MS for Autonomy - Experiment Launcher
REM Double-click this file to launch the experiment menu
REM ============================================================

REM Change to this script's directory
cd /d "%~dp0"

REM Use the isaaclab Python directly (C:\miniconda3\envs\isaaclab)
"C:\miniconda3\envs\isaaclab311\python.exe" launch.py %*

REM Keep window open so you can read output
pause
