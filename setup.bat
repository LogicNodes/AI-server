@echo off
REM =============================================================================
REM AI-Server Setup Script for Windows
REM Automated installation and configuration
REM =============================================================================

title AI-Server Setup - Windows
color 0A

echo.
echo  ===============================================
echo  🚀 AI-Server Setup Script - Windows
echo  ===============================================
echo.
echo  This script will set up your AI-Server system
echo  Please ensure you have administrator rights
echo.
pause

REM Check if running as administrator
net session >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: This script must be run as Administrator
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo ✅ Running with administrator privileges
echo.

REM =============================================================================
REM Step 1: Check System Requirements
REM =============================================================================

echo 📋 Step 1: Checking system requirements...
echo.

REM Check Windows version
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j
echo Windows Version: %VERSION%

if "%VERSION%" LSS "10.0" (
    echo ❌ ERROR: Windows 10 or later required
    pause
    exit /b 1
)

echo ✅ Windows version supported

REM Check available memory
for /f "skip=1" %%p in ('wmic computersystem get TotalPhysicalMemory') do (
    set RAM_BYTES=%%p
    goto :break_ram
)
:break_ram
set /a RAM_GB=%RAM_BYTES:~0,-9%
echo System RAM: %RAM_GB%GB

if %RAM_GB% LSS 8 (
    echo ⚠️  WARNING: 8GB+ RAM recommended, you have %RAM_GB%GB
) else (
    echo ✅ Sufficient RAM available
)

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ❌ WARNING: NVIDIA GPU or drivers not detected
    echo Please install NVIDIA drivers from nvidia.com
    echo.
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
) else (
    echo ✅ NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

echo.

REM =============================================================================
REM Step 2: Install Docker Desktop
REM =============================================================================

echo 📦 Step 2: Installing Docker Desktop...
echo.

docker --version >nul 2>&1
if errorlevel 1 (
    echo Docker not found. Installing Docker Desktop...
    
    REM Download Docker Desktop installer
    echo Downloading Docker Desktop installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://desktop.docker.com/win/main/amd64/Docker%%20Desktop%%20Installer.exe' -OutFile 'DockerDesktopInstaller.exe'"
    
    if not exist "DockerDesktopInstaller.exe" (
        echo ❌ ERROR: Failed to download Docker Desktop
        echo Please download manually from docker.com
        pause
        exit /b 1
    )
    
    echo Installing Docker Desktop (this may take several minutes)...
    start /wait DockerDesktopInstaller.exe install --quiet
    
    echo Docker Desktop installed. Please:
    echo 1. Restart your computer
    echo 2. Start Docker Desktop from Start Menu
    echo 3. Run this setup script again
    echo.
    del DockerDesktopInstaller.exe >nul 2>&1
    pause
    exit /b 0
) else (
    echo ✅ Docker Desktop already installed
    docker --version
)

REM Check if Docker daemon is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Desktop is not running
    echo Please start Docker Desktop and run this script again
    pause
    exit /b 1
)

echo ✅ Docker Desktop is running

REM Check for GPU support in Docker
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️  GPU support not available in Docker
    echo Please enable GPU support in Docker Desktop settings
    echo Go to Settings > Resources > WSL Integration
) else (
    echo ✅ Docker GPU support available
)

echo.

REM =============================================================================
REM Step 3: Install Python
REM =============================================================================

echo 🐍 Step 3: Checking Python installation...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Installing Python...
    
    REM Download Python installer
    echo Downloading Python 3.11 installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe' -OutFile 'python-installer.exe'"
    
    if not exist "python-installer.exe" (
        echo ❌ ERROR: Failed to download Python
        echo Please install Python manually from python.org
        pause
        exit /b 1
    )
    
    echo Installing Python 3.11...
    start /wait python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    
    del python-installer.exe >nul 2>&1
    
    REM Refresh PATH
    call refreshenv >nul 2>&1
    
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Python installation failed
        echo Please install Python manually and restart this script
        pause
        exit /b 1
    )
) else (
    echo ✅ Python already installed
)

python --version
pip --version

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

echo.

REM =============================================================================
REM Step 4: Setup Project Structure
REM =============================================================================

echo 📁 Step 4: Setting up project structure...
echo.

REM Create directories
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist ".cache" mkdir .cache

echo ✅ Created project directories

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating default configuration...
    (
        echo MODEL_NAME=qwen-2.5-1.5b-instruct
        echo API_KEY=hardq_dev_key_001
        echo MAX_CONCURRENT=2
        echo MAX_TOKENS_LIMIT=1000
        echo CUDA_VISIBLE_DEVICES=0
    ) > .env
    echo ✅ Created .env configuration file
) else (
    echo ✅ Configuration file already exists
)

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install some Python dependencies
    echo This may not affect Docker operation
)

echo.

REM =============================================================================
REM Step 5: Build Docker Image
REM =============================================================================

echo 🐳 Step 5: Building Docker image...
echo.

echo This will take 10-15 minutes to download and build...
echo Building optimized Docker image with CUDA support...

docker build -t llm-server:latest .

if errorlevel 1 (
    echo ❌ Docker image build failed
    echo Please check the error messages above
    pause
    exit /b 1
)

echo ✅ Docker image built successfully

REM Test the image
echo Testing Docker image...
docker run --rm llm-server:latest python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo.

REM =============================================================================
REM Step 6: Download Sample Model (Optional)
REM =============================================================================

echo 🧠 Step 6: Download sample model (optional)...
echo.

if not exist "models\qwen-2.5-1.5b-instruct" (
    set /p download="Download Qwen 2.5 1.5B model (~3GB)? (y/N): "
    if /i "%download%"=="y" (
        echo Downloading model... this may take 10-20 minutes
        python -c "
from huggingface_hub import snapshot_download
import os
os.makedirs('models', exist_ok=True)
try:
    snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/qwen-2.5-1.5b-instruct')
    print('✅ Model downloaded successfully')
except Exception as e:
    print('❌ Model download failed:', e)
"
    )
) else (
    echo ✅ Model already exists
)

echo.

REM =============================================================================
REM Step 7: Final Setup and Testing
REM =============================================================================

echo 🧪 Step 7: Final setup and testing...
echo.

REM Create desktop shortcut
echo Creating desktop shortcut...
set DESKTOP=%USERPROFILE%\Desktop
set SHORTCUT=%DESKTOP%\AI-Server.lnk
set TARGET=%CD%\start.bat

powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT%'); $Shortcut.TargetPath = '%TARGET%'; $Shortcut.WorkingDirectory = '%CD%'; $Shortcut.Description = 'AI-Server Management Interface'; $Shortcut.Save()"

echo ✅ Desktop shortcut created

REM Test basic functionality
echo Testing basic functionality...

REM Check if CLI runs
python cli.py --help >nul 2>&1
if errorlevel 1 (
    echo ⚠️  CLI test failed - may have minor issues
) else (
    echo ✅ CLI interface working
)

echo.

REM =============================================================================
REM Setup Complete
REM =============================================================================

echo  ===============================================
echo  🎉 AI-Server Setup Complete!
echo  ===============================================
echo.
echo  ✅ Docker Desktop installed and configured
echo  ✅ Python environment ready
echo  ✅ Docker image built (llm-server:latest)
echo  ✅ Project structure created
echo  ✅ Desktop shortcut created
echo.
echo  📋 Next Steps:
echo.
echo  1. Double-click "AI-Server" shortcut on desktop
echo     OR run: start.bat
echo.
echo  2. From the menu, choose:
echo     - Option 7: List Available Models
echo     - Option 3: Start Server Container
echo.
echo  3. Access your API at: http://localhost:8080
echo     API Key: hardq_dev_key_001
echo.
echo  📖 Documentation:
echo     - README.md for detailed usage
echo     - test.py for API examples
echo.
echo  🔧 Troubleshooting:
echo     - Check Docker Desktop is running
echo     - Ensure NVIDIA drivers are updated
echo     - View logs with Option 6 in menu
echo.

pause

REM Optional: Start the system
set /p start_now="Start AI-Server now? (y/N): "
if /i "%start_now%"=="y" (
    echo.
    echo Starting AI-Server management interface...
    call start.bat
)

echo.
echo Setup script finished. You can run this script again anytime to update.
pause