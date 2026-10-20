@echo off
cd /d "%~dp0"
if not exist .hosting mkdir .hosting
echo stop>.hosting\stop
echo Sharing will stop in a moment. Local translation stays running.
