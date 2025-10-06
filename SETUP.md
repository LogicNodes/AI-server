# 🚀 AI-Server Setup Scripts

## Quick Setup for New Systems

Your AI-Server now includes automated setup scripts for both Windows and Linux systems. These scripts will install all dependencies, configure the environment, and get your AI server running in one go.

## 🪟 Windows Setup

### Prerequisites
- Windows 10/11
- Administrator privileges
- Internet connection

### Installation
1. **Right-click `setup.bat`** and select **"Run as administrator"**
2. **Follow the prompts** - the script will:
   - ✅ Check system requirements
   - ✅ Install Docker Desktop with GPU support
   - ✅ Install Python 3.11
   - ✅ Build the Docker image (~15 minutes)
   - ✅ Create desktop shortcut
   - ✅ Optionally download a sample model

### What Gets Installed
- **Docker Desktop** (latest version with GPU support)
- **Python 3.11** with pip
- **NVIDIA Container Toolkit** (if NVIDIA GPU detected)
- **All Python dependencies** from requirements.txt
- **Desktop shortcut** to AI-Server

### After Setup
- **Double-click** the "AI-Server" desktop shortcut
- **Or run**: `start.bat`

---

## 🐧 Linux Setup

### Prerequisites
- Ubuntu 20.04+, Debian 11+, Fedora 35+, or CentOS 8+
- Sudo privileges
- Internet connection

### Installation
```bash
# Make the script executable and run it
chmod +x setup.sh
./setup.sh
```

### What Gets Installed
- **NVIDIA Drivers** (if NVIDIA GPU detected)
- **Docker Engine** with GPU support
- **NVIDIA Container Toolkit**
- **Python 3** with pip and dev tools
- **All dependencies** and project structure
- **Desktop shortcut** (if in desktop environment)

### After Setup
```bash
# Start the management interface
./start.sh

# Or directly
python3 cli.py
```

---

## 📋 What Both Scripts Do

### 1. **System Requirements Check**
- RAM (8GB+ recommended)
- Disk space (50GB+ recommended)
- GPU detection (NVIDIA GPUs)

### 2. **Dependency Installation**
- Docker with GPU support
- Python environment
- NVIDIA drivers (if needed)
- All required packages

### 3. **Project Setup**
- Create directory structure (`models/`, `logs/`)
- Generate `.env` configuration file
- Build Docker image with CUDA support
- Install Python dependencies

### 4. **Optional Components**
- Download sample model (Qwen 2.5 1.5B ~3GB)
- Create desktop shortcuts
- Test all components

### 5. **Final Testing**
- Verify Docker image works
- Test CLI interface
- Check GPU support
- Validate API endpoints

---

## 🎯 Expected Installation Time

| Component | Windows | Linux |
|-----------|---------|-------|
| Dependencies | 10-15 min | 5-10 min |
| Docker Image Build | 10-15 min | 10-15 min |
| Model Download | 10-20 min | 10-20 min |
| **Total** | **30-50 min** | **25-45 min** |

*Times vary based on internet speed and system performance*

---

## 🔧 Troubleshooting

### Common Issues

**"Docker not found" (Windows)**
```bash
# The setup script will auto-install Docker Desktop
# You may need to restart Windows after Docker installation
```

**"Permission denied" (Linux)**
```bash
# Ensure the script is executable
chmod +x setup.sh

# Run with proper user (not root)
./setup.sh
```

**"NVIDIA drivers not found"**
```bash
# Windows: Download from nvidia.com
# Linux: The script will attempt to install them
```

**"Docker build failed"**
```bash
# Check internet connection
# Ensure sufficient disk space (50GB+)
# Try running the setup script again
```

### Manual Recovery

If the setup script fails, you can run individual steps:

```bash
# Build just the Docker image
docker build -t llm-server:latest .

# Test the CLI
python3 cli.py

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
```

---

## 🎉 After Successful Setup

Your system will have:

- ✅ **Fully configured AI server** ready to run
- ✅ **Docker image** with all dependencies (~22GB)
- ✅ **Management interface** accessible via start script
- ✅ **GPU acceleration** (if NVIDIA GPU present)
- ✅ **OpenAI-compatible API** server
- ✅ **Model storage** system ready for any Hugging Face model

### Next Steps:
1. **Run** `start.bat` (Windows) or `./start.sh` (Linux)
2. **Choose Option 7** to see available models
3. **Choose Option 3** to start your AI server
4. **Access** your API at `http://localhost:8080`

---

## 🔄 Re-running Setup

You can safely run the setup scripts multiple times to:
- Update dependencies
- Rebuild Docker images
- Fix configuration issues
- Install missing components

The scripts are designed to be **idempotent** - they won't break existing installations.

---

## 📞 Support

If you encounter issues:

1. **Check the output** of the setup script for specific error messages
2. **Ensure prerequisites** are met (admin rights, internet, etc.)
3. **Try running again** - many issues are transient
4. **Check Docker Desktop** is running (Windows)
5. **Verify NVIDIA drivers** with `nvidia-smi`

The setup scripts include comprehensive error checking and will guide you through most common issues.