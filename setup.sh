#!/bin/bash
# =============================================================================
# AI-Server Setup Script for Linux
# Automated installation and configuration
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emojis (will work on most modern Linux terminals)
CHECK="✅"
ERROR="❌"
WARNING="⚠️ "
INFO="📋"
ROCKET="🚀"
PACKAGE="📦"
PYTHON="🐍"
DOCKER="🐳"
BRAIN="🧠"
TEST="🧪"
PARTY="🎉"

print_header() {
    echo -e "${BLUE}"
    echo "  ==============================================="
    echo "  ${ROCKET} AI-Server Setup Script - Linux"
    echo "  ==============================================="
    echo -e "${NC}"
    echo "  This script will set up your AI-Server system"
    echo "  Please ensure you have sudo privileges"
    echo
}

print_step() {
    echo -e "${PURPLE}$1${NC}"
    echo
}

print_success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

print_error() {
    echo -e "${RED}${ERROR} $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
}

print_info() {
    echo -e "${CYAN}${INFO} $1${NC}"
}

# Check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_error "Please don't run this script as root"
        print_info "Run as regular user with sudo privileges"
        exit 1
    fi
    print_success "Running as regular user"
}

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    else
        print_error "Cannot detect Linux distribution"
        exit 1
    fi
    
    print_info "Detected: $PRETTY_NAME"
    
    # Set package manager commands
    case $DISTRO in
        ubuntu|debian)
            PKG_UPDATE="sudo apt update"
            PKG_INSTALL="sudo apt install -y"
            PKG_UPGRADE="sudo apt upgrade -y"
            ;;
        fedora)
            PKG_UPDATE="sudo dnf check-update || true"
            PKG_INSTALL="sudo dnf install -y"
            PKG_UPGRADE="sudo dnf upgrade -y"
            ;;
        centos|rhel)
            PKG_UPDATE="sudo yum check-update || true"
            PKG_INSTALL="sudo yum install -y"
            PKG_UPGRADE="sudo yum upgrade -y"
            ;;
        *)
            print_warning "Unsupported distribution: $DISTRO"
            print_info "This script supports Ubuntu, Debian, Fedora, CentOS, RHEL"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
            ;;
    esac
}

# Check system requirements
check_requirements() {
    print_step "${INFO} Step 1: Checking system requirements..."
    
    # Check memory
    RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    RAM_GB=$((RAM_KB / 1024 / 1024))
    
    print_info "System RAM: ${RAM_GB}GB"
    
    if [ $RAM_GB -lt 8 ]; then
        print_warning "8GB+ RAM recommended, you have ${RAM_GB}GB"
    else
        print_success "Sufficient RAM available"
    fi
    
    # Check disk space
    DISK_AVAIL=$(df . | tail -1 | awk '{print $4}')
    DISK_GB=$((DISK_AVAIL / 1024 / 1024))
    
    print_info "Available disk space: ${DISK_GB}GB"
    
    if [ $DISK_GB -lt 50 ]; then
        print_warning "50GB+ disk space recommended, you have ${DISK_GB}GB"
    else
        print_success "Sufficient disk space available"
    fi
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    else
        print_warning "NVIDIA GPU or drivers not detected"
        print_info "Please install NVIDIA drivers if you have an NVIDIA GPU"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    echo
}

# Update system packages
update_system() {
    print_step "${PACKAGE} Step 2: Updating system packages..."
    
    print_info "Updating package lists..."
    $PKG_UPDATE
    
    print_info "Upgrading existing packages..."
    $PKG_UPGRADE
    
    print_success "System packages updated"
    echo
}

# Install NVIDIA drivers
install_nvidia_drivers() {
    print_step "${PACKAGE} Step 3: Installing NVIDIA drivers..."
    
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA drivers already installed"
        nvidia-smi --version | head -1
    else
        print_info "Installing NVIDIA drivers..."
        
        case $DISTRO in
            ubuntu|debian)
                $PKG_INSTALL nvidia-driver-525 nvidia-utils-525
                ;;
            fedora)
                # Enable RPM Fusion
                sudo dnf install -y https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
                sudo dnf install -y https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
                $PKG_INSTALL akmod-nvidia xorg-x11-drv-nvidia-cuda
                ;;
            centos|rhel)
                # Enable EPEL
                $PKG_INSTALL epel-release
                # Install NVIDIA drivers (requires manual intervention)
                print_warning "Please install NVIDIA drivers manually for CentOS/RHEL"
                print_info "Visit: https://developer.nvidia.com/cuda-downloads"
                ;;
        esac
        
        print_warning "Please reboot your system and run this script again after installing NVIDIA drivers"
        read -p "Reboot now? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo reboot
        else
            print_info "Please reboot manually and re-run this script"
            exit 0
        fi
    fi
    
    echo
}

# Install Docker
install_docker() {
    print_step "${DOCKER} Step 4: Installing Docker..."
    
    if command -v docker &> /dev/null; then
        print_success "Docker already installed"
        docker --version
    else
        print_info "Installing Docker..."
        
        # Download and run Docker installation script
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        rm get-docker.sh
        
        # Add user to docker group
        sudo usermod -aG docker $USER
        
        print_success "Docker installed successfully"
        print_warning "Please log out and log back in for Docker group membership to take effect"
        
        # Try to start Docker service
        sudo systemctl enable docker
        sudo systemctl start docker
    fi
    
    # Test Docker
    if docker info &> /dev/null; then
        print_success "Docker daemon is running"
    else
        print_warning "Docker daemon not running, trying to start..."
        sudo systemctl start docker
        sleep 3
        
        if docker info &> /dev/null; then
            print_success "Docker daemon started"
        else
            print_error "Failed to start Docker daemon"
            print_info "Please start Docker manually: sudo systemctl start docker"
            exit 1
        fi
    fi
    
    echo
}

# Install NVIDIA Container Toolkit
install_nvidia_docker() {
    print_step "${DOCKER} Step 5: Installing NVIDIA Container Toolkit..."
    
    # Test if nvidia-docker is already working
    if docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        print_success "NVIDIA Container Toolkit already working"
    else
        print_info "Installing NVIDIA Container Toolkit..."
        
        case $DISTRO in
            ubuntu|debian)
                # Add NVIDIA package repository
                distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
                curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
                curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
                
                $PKG_UPDATE
                $PKG_INSTALL nvidia-docker2
                ;;
            fedora)
                curl -s -L https://nvidia.github.io/nvidia-docker/centos8/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
                $PKG_INSTALL nvidia-docker2
                ;;
            centos|rhel)
                distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed s/centos/centos/g | sed s/rhel/centos/g)
                curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
                $PKG_INSTALL nvidia-docker2
                ;;
        esac
        
        # Restart Docker
        sudo systemctl restart docker
        
        # Test NVIDIA Docker
        print_info "Testing NVIDIA Docker support..."
        if docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
            print_success "NVIDIA Container Toolkit working correctly"
        else
            print_warning "NVIDIA Container Toolkit test failed"
            print_info "This may work after a reboot"
        fi
    fi
    
    echo
}

# Install Python
install_python() {
    print_step "${PYTHON} Step 6: Installing Python..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python already installed: $PYTHON_VERSION"
    else
        print_info "Installing Python 3..."
        $PKG_INSTALL python3 python3-pip python3-dev
    fi
    
    # Install additional tools
    $PKG_INSTALL git curl wget
    
    # Upgrade pip
    print_info "Upgrading pip..."
    python3 -m pip install --upgrade pip --user
    
    print_success "Python environment ready"
    python3 --version
    pip3 --version
    
    echo
}

# Setup project structure
setup_project() {
    print_step "${INFO} Step 7: Setting up project structure..."
    
    # Create directories
    mkdir -p models logs .cache
    print_success "Created project directories"
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        print_info "Creating default configuration..."
        cat > .env << EOF
MODEL_NAME=qwen-2.5-1.5b-instruct
API_KEY=hardq_dev_key_001
MAX_CONCURRENT=2
MAX_TOKENS_LIMIT=1000
CUDA_VISIBLE_DEVICES=0
EOF
        print_success "Created .env configuration file"
    else
        print_success "Configuration file already exists"
    fi
    
    # Set proper permissions
    chmod +x start.sh cli.py 2>/dev/null || true
    chmod 755 models/ logs/ .cache/
    
    # Install Python dependencies
    if [ -f requirements.txt ]; then
        print_info "Installing Python dependencies..."
        pip3 install -r requirements.txt --user
        if [ $? -eq 0 ]; then
            print_success "Python dependencies installed"
        else
            print_warning "Some Python dependencies failed to install"
            print_info "This may not affect Docker operation"
        fi
    fi
    
    echo
}

# Build Docker image
build_docker_image() {
    print_step "${DOCKER} Step 8: Building Docker image..."
    
    if [ ! -f Dockerfile ]; then
        print_error "Dockerfile not found in current directory"
        print_info "Please ensure you're in the AI-server directory"
        exit 1
    fi
    
    print_info "This will take 10-15 minutes to download and build..."
    print_info "Building optimized Docker image with CUDA support..."
    
    if docker build -t llm-server:latest .; then
        print_success "Docker image built successfully"
        
        # Test the image
        print_info "Testing Docker image..."
        docker run --rm llm-server:latest python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
    else
        print_error "Docker image build failed"
        print_info "Please check the error messages above"
        exit 1
    fi
    
    echo
}

# Download sample model
download_model() {
    print_step "${BRAIN} Step 9: Download sample model (optional)..."
    
    if [ ! -d "models/qwen-2.5-1.5b-instruct" ]; then
        read -p "Download Qwen 2.5 1.5B model (~3GB)? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Downloading model... this may take 10-20 minutes"
            
            python3 -c "
from huggingface_hub import snapshot_download
import os
os.makedirs('models', exist_ok=True)
try:
    snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/qwen-2.5-1.5b-instruct')
    print('${CHECK} Model downloaded successfully')
except Exception as e:
    print('${ERROR} Model download failed:', e)
" || print_warning "Model download failed - you can download it later using the CLI"
        fi
    else
        print_success "Model already exists"
    fi
    
    echo
}

# Create desktop shortcut (if in desktop environment)
create_shortcuts() {
    print_step "${INFO} Step 10: Creating shortcuts..."
    
    # Create start.sh if it doesn't exist
    if [ ! -f start.sh ]; then
        print_info "Creating start.sh script..."
        cat > start.sh << 'EOF'
#!/bin/bash
# AI-Server Linux Launcher

echo "🐧 AI-Server Management System - Linux"
echo "========================================"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running"
    echo "Please start Docker: sudo systemctl start docker"
    exit 1
fi

echo "✅ All prerequisites satisfied"
echo "Starting management interface..."
echo

python3 cli.py

if [ $? -ne 0 ]; then
    echo
    echo "An error occurred. Press Enter to exit."
    read
fi
EOF
        chmod +x start.sh
        print_success "Created start.sh launcher"
    fi
    
    # Try to create desktop shortcut if in desktop environment
    if [ -n "$DESKTOP_SESSION" ] && [ -d "$HOME/Desktop" ]; then
        DESKTOP_FILE="$HOME/Desktop/AI-Server.desktop"
        cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Name=AI-Server
Comment=AI-Server Management Interface
Exec=$PWD/start.sh
Icon=applications-internet
Terminal=true
Type=Application
Categories=Development;
Path=$PWD
EOF
        chmod +x "$DESKTOP_FILE"
        print_success "Created desktop shortcut"
    fi
    
    echo
}

# Final testing
final_testing() {
    print_step "${TEST} Step 11: Final testing..."
    
    # Test CLI
    if python3 cli.py --help &> /dev/null; then
        print_success "CLI interface working"
    else
        print_warning "CLI test failed - may have minor issues"
    fi
    
    # Test Docker image
    if docker images | grep -q llm-server; then
        print_success "Docker image available"
    else
        print_warning "Docker image not found"
    fi
    
    echo
}

# Print completion message
print_completion() {
    echo -e "${GREEN}"
    echo "  ==============================================="
    echo "  ${PARTY} AI-Server Setup Complete!"
    echo "  ==============================================="
    echo -e "${NC}"
    echo
    echo -e "${GREEN}${CHECK}${NC} Docker installed and configured"
    echo -e "${GREEN}${CHECK}${NC} Python environment ready"
    echo -e "${GREEN}${CHECK}${NC} Docker image built (llm-server:latest)"
    echo -e "${GREEN}${CHECK}${NC} Project structure created"
    echo
    echo -e "${BLUE}${INFO} Next Steps:${NC}"
    echo
    echo "  1. Run the management interface:"
    echo "     ./start.sh"
    echo
    echo "  2. From the menu, choose:"
    echo "     - Option 7: List Available Models"
    echo "     - Option 3: Start Server Container"
    echo
    echo "  3. Access your API at: http://localhost:8080"
    echo "     API Key: hardq_dev_key_001"
    echo
    echo -e "${BLUE}${INFO} Documentation:${NC}"
    echo "     - README.md for detailed usage"
    echo "     - test.py for API examples"
    echo
    echo -e "${BLUE}${INFO} Troubleshooting:${NC}"
    echo "     - Ensure Docker daemon is running: sudo systemctl start docker"
    echo "     - Check NVIDIA drivers: nvidia-smi"
    echo "     - View logs with Option 6 in menu"
    echo
}

# Main installation flow
main() {
    print_header
    
    check_root
    detect_distro
    check_requirements
    update_system
    install_nvidia_drivers
    install_docker
    install_nvidia_docker
    install_python
    setup_project
    build_docker_image
    download_model
    create_shortcuts
    final_testing
    print_completion
    
    # Optional: Start the system
    read -p "Start AI-Server now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo
        print_info "Starting AI-Server management interface..."
        ./start.sh
    fi
    
    echo
    print_info "Setup script finished. You can run this script again anytime to update."
}

# Run main function
main "$@"