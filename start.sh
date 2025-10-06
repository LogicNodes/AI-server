#!/bin/bash
# =============================================================================
# AI-Server Launcher - Linux
# Model management and server operations
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_header() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "         AI-SERVER MANAGEMENT SYSTEM - LINUX"
    echo "============================================================"
    echo -e "${NC}"
}

check_prerequisites() {
    local all_good=true

    print_info "Checking prerequisites..."

    # Check Python
    if command -v python3 &> /dev/null; then
        print_success "Python found: $(python3 --version)"
    else
        print_error "Python 3 not found"
        echo "Please install Python 3: sudo apt install python3 python3-pip"
        all_good=false
    fi

    # Check Docker
    if command -v docker &> /dev/null; then
        print_success "Docker found: $(docker --version)"

        # Check if Docker daemon is running
        if docker info &> /dev/null; then
            print_success "Docker daemon is running"
        else
            print_error "Docker daemon is not running"
            echo "Please start Docker: sudo systemctl start docker"
            all_good=false
        fi
    else
        print_error "Docker not found"
        echo "Please run setup.sh first to install Docker"
        all_good=false
    fi

    # Check NVIDIA GPU (optional)
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1

        # Test GPU access in Docker
        if docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
            print_success "Docker GPU support working"
        else
            print_warning "Docker GPU support not working"
            echo "Models will run on CPU (slower)"
        fi
    else
        print_warning "NVIDIA GPU not detected"
        echo "Models will run on CPU only"
    fi

    # Check Python dependencies
    if python3 -c "import requests, fastapi" &> /dev/null; then
        print_success "Python dependencies available"
    else
        print_warning "Some Python dependencies missing"
        echo "Installing missing dependencies..."
        pip3 install -r requirements.txt --user
    fi

    if [ "$all_good" = false ]; then
        echo
        print_error "Prerequisites not met. Please fix the issues above."
        echo "If this is a fresh system, run: sudo ./setup.sh"
        exit 1
    fi

    print_success "All prerequisites satisfied"
    echo
}

main() {
    clear
    print_header

    check_prerequisites

    print_info "Starting management interface..."
    echo

    # Run the CLI interface
    python3 cli.py

    # Check exit status
    if [ $? -ne 0 ]; then
        echo
        print_error "An error occurred in the management interface"
        echo "Check the error messages above for details"
        echo
        read -p "Press Enter to exit..."
    fi
}

# Make sure we're in the right directory
if [ ! -f "cli.py" ]; then
    print_error "CLI interface not found!"
    echo "Please run this script from the AI-server directory"
    exit 1
fi

# Run main function
main "$@"