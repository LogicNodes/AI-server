@echo off
title LLM Server Management System

echo Checking Docker installation...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not running
    echo Please install Docker Desktop from docker.com
    echo Make sure Docker Desktop is running before continuing
    pause
    exit /b 1
)

echo Docker found! Starting management interface...
echo.

python cli.py

if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit.
    pause >nul
)