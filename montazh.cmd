@echo off
rem Run the montazh skill from this folder: montazh.cmd auto --src "material" ...
rem ASCII only in this file: cmd.exe reads it in the console codepage.
setlocal
pushd "%~dp0"
python -m montazh %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
