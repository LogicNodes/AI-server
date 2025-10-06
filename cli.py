#!/usr/bin/env python3
"""
LLM Server Management CLI
Clean, efficient container management
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
from model_manager import ModelManager

class LLMManager:
    """LLM server management"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.api_key = "hardq_dev_key_001"
        self.image_name = "llm-server:latest"
        self.container_name = "llm-server"
    
    def main_menu(self):
        """Main interface menu"""
        while True:
            self.clear_screen()
            self.print_header()
            
            print("LLM Server Management")
            print()
            print("1. Download New Model")
            print("2. Build Docker Image") 
            print("3. Start Server Container")
            print("4. Stop Server Container")
            print("5. View Server Status")
            print("6. View Server Logs")
            print("7. List Available Models")
            print("8. Exit")
            print()
            
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1':
                self.download_model_menu()
            elif choice == '2':
                self.build_image_menu()
            elif choice == '3':
                self.start_server_menu()
            elif choice == '4':
                self.stop_server()
            elif choice == '5':
                self.view_status()
            elif choice == '6':
                self.view_logs()
            elif choice == '7':
                self.list_models_menu()
            elif choice == '8':
                print("Goodbye!")
                break
            else:
                input("Invalid choice. Press Enter to continue...")
    
    def download_model_menu(self):
        """Download model interface"""
        self.clear_screen()
        print("=== DOWNLOAD NEW MODEL ===")
        print()
        print("Enter a Hugging Face model URL or model ID:")
        print("Examples:")
        print("  microsoft/DialoGPT-small")
        print("  Qwen/Qwen2.5-3B-Instruct")
        print("  meta-llama/Llama-2-7b-chat-hf")
        print()
        
        url = input("Model URL or ID: ").strip()
        if not url:
            return
        
        if not url.startswith('http'):
            url = f"https://huggingface.co/{url}"
        
        print()
        print("Downloading model...")
        success, message = self.model_manager.download_model(url)
        
        print()
        if success:
            print(f"[OK] {message}")
            print()
            print("[TIP] Tip: Run 'Build Docker Image' to update the container with new models")
        else:
            print(f"[ERROR] {message}")
        
        input("Press Enter to continue...")
    
    def build_image_menu(self):
        """Build Docker image"""
        self.clear_screen()
        print("=== BUILD DOCKER IMAGE ===")
        print()
        print("This will build a Docker image with all dependencies pre-installed.")
        print("The image will be ~5GB and take 10-15 minutes to build.")
        print()
        
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            return
        
        print()
        print("Building Docker image...")
        print("This may take a while - perfect time for coffee! ")
        print()
        
        success = self.build_docker_image()
        
        print()
        if success:
            print("[OK] Docker image built successfully!")
            print(f"   Image: {self.image_name}")
        else:
            print("[ERROR] Docker image build failed")
            print("   Check Docker is installed and running")
        
        input("Press Enter to continue...")
    
    def start_server_menu(self):
        """Start server container"""
        self.clear_screen()
        print("=== START SERVER CONTAINER ===")
        print()
        
        # Check if image exists
        if not self.check_image_exists():
            print("[ERROR] Docker image not found!")
            print("   Run 'Build Docker Image' first")
            input("Press Enter to continue...")
            return
        
        # List available models
        models = self.model_manager.list_available_models()
        if not models:
            print("[ERROR] No models available!")
            print("   Download a model first")
            input("Press Enter to continue...")
            return
        
        print("Available models:")
        print()
        
        for i, model in enumerate(models, 1):
            compatible, msg, stats = self.model_manager.validate_gpu_compatibility(model['name'])
            status = "[OK]" if compatible else "[ERROR]"
            
            print(f"{i}. {model['name']} {status}")
            print(f"   Parameters: {model['params']}")
            print(f"   VRAM: {model['vram_gb']}GB")
            print(f"   Status: {msg}")
            print()
        
        try:
            choice = int(input(f"Select model (1-{len(models)}): ")) - 1
            if 0 <= choice < len(models):
                selected_model = models[choice]
                self.start_server_container(selected_model)
            else:
                print("Invalid selection.")
                input("Press Enter to continue...")
        except ValueError:
            print("Invalid input.")
            input("Press Enter to continue...")
    
    def start_server_container(self, model: Dict[str, str]):
        """Start the server container with selected model"""
        self.clear_screen()
        print("=== STARTING CONTAINER ===")
        print()
        
        # Stop existing container if running
        self.stop_container_silent()
        
        model_name = model['name']
        max_concurrent = 2  # Conservative for most GPUs
        
        print(f"Starting container with model: {model_name}")
        print(f"Max concurrent requests: {max_concurrent}")
        print(f"API Key: {self.api_key}")
        print()
        
        # Create environment file
        env_content = f"""MODEL_NAME={model_name}
API_KEY={self.api_key}
MAX_CONCURRENT={max_concurrent}
MAX_TOKENS_LIMIT=1000"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("Starting Docker container...")
        success = self.start_docker_container()
        
        if success:
            print("[OK] Container started successfully!")
            print()
            print(" Server starting up... (may take 1-2 minutes to load model)")
            print(" API available at: http://localhost:8080")
            print(f" API Key: {self.api_key}")
            print()
            print("Waiting for server to be ready...")
            
            # Wait for server to be ready
            if self.wait_for_server_ready():
                print("[OK] Server is ready!")
                print()
                print("Available endpoints:")
                print("  GET  /              - Server status")
                print("  GET  /health        - Health check")
                print("  GET  /v1/models     - List models")
                print("  POST /v1/chat/completions - Chat API")
                print()
                print("Press Enter to return to menu (server keeps running)...")
            else:
                print("[ERROR] Server failed to start properly")
                print("   Check logs with 'View Server Logs'")
        else:
            print("[ERROR] Failed to start container")
            print("   Make sure Docker is running and has GPU access")
        
        input("Press Enter to continue...")
    
    def stop_server(self):
        """Stop the server container"""
        self.clear_screen()
        print("=== STOP SERVER CONTAINER ===")
        print()
        
        if self.is_container_running():
            print("Stopping server container...")
            success = self.stop_docker_container()
            
            if success:
                print("[OK] Container stopped successfully!")
            else:
                print("[ERROR] Failed to stop container")
        else:
            print("[INFO]  No server container is currently running")
        
        input("Press Enter to continue...")
    
    def view_status(self):
        """View server status"""
        self.clear_screen()
        print("=== SERVER STATUS ===")
        print()
        
        if self.is_container_running():
            print(" Container Status:")
            self.show_container_status()
            print()
            
            print(" API Status:")
            self.show_api_status()
        else:
            print("[INFO]  Server container is not running")
        
        input("\nPress Enter to continue...")
    
    def view_logs(self):
        """View server logs"""
        self.clear_screen()
        print("=== SERVER LOGS ===")
        print()
        
        if self.is_container_running():
            print(" Recent logs (last 50 lines):")
            print("-" * 60)
            self.show_container_logs()
        else:
            print("[INFO]  Server container is not running")
        
        print()
        input("Press Enter to continue...")
    
    def list_models_menu(self):
        """List available models"""
        self.clear_screen()
        print("=== AVAILABLE MODELS ===")
        print()
        
        models = self.model_manager.list_available_models()
        if not models:
            print("No models available.")
            print("Use 'Download New Model' to add models.")
        else:
            for i, model in enumerate(models, 1):
                compatible, msg, _ = self.model_manager.validate_gpu_compatibility(model['name'])
                status = "[OK] Ready" if compatible else "[ERROR] Incompatible"
                
                print(f"{i}. {model['name']}")
                print(f"   Parameters: {model['params']}")
                print(f"   VRAM Required: {model['vram_gb']}GB")
                print(f"   Disk Size: {model['size_gb']}GB")
                print(f"   Status: {status}")
                print(f"   Details: {msg}")
                print()
        
        input("Press Enter to continue...")
    
    # Docker operations
    def check_docker_available(self) -> bool:
        """Check if Docker is available"""
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def check_image_exists(self) -> bool:
        """Check if Docker image exists"""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.image_name],
                capture_output=True, text=True
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False
    
    def build_docker_image(self) -> bool:
        """Build Docker image"""
        try:
            result = subprocess.run(
                ["docker", "build", "-t", self.image_name, "."],
                capture_output=False  # Show output to user
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Build failed: {e}")
            return False
    
    def start_docker_container(self) -> bool:
        """Start Docker container using docker-compose"""
        try:
            result = subprocess.run(
                ["docker-compose", "up", "-d"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Start failed: {e}")
            return False
    
    def stop_docker_container(self) -> bool:
        """Stop Docker container"""
        try:
            result = subprocess.run(
                ["docker-compose", "down"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def stop_container_silent(self):
        """Stop container without error messages"""
        try:
            subprocess.run(["docker-compose", "down"], capture_output=True)
        except Exception:
            pass
    
    def is_container_running(self) -> bool:
        """Check if container is running"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )
            return self.container_name in result.stdout
        except Exception:
            return False
    
    def show_container_status(self):
        """Show container status"""
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", 
                 "table {{.Container}}\\t{{.CPUPerc}}\\t{{.MemUsage}}\\t{{.MemPerc}}", 
                 self.container_name],
                capture_output=True, text=True
            )
            print(result.stdout)
        except Exception as e:
            print(f"Could not get container stats: {e}")
    
    def show_container_logs(self):
        """Show container logs"""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", "50", self.container_name],
                capture_output=True, text=True
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
        except Exception as e:
            print(f"Could not get logs: {e}")
    
    def show_api_status(self):
        """Show API status"""
        try:
            import requests
            response = requests.get("http://localhost:8080/", timeout=10)
            if response.status_code == 200:
                status = response.json()
                print(f"[OK] API responding")
                print(f"   Model: {status['model']['name']} ({'loaded' if status['model']['loaded'] else 'not loaded'})")
                print(f"   GPU: {status['gpu'].get('memory_used_gb', 0):.1f}GB used")
                print(f"   Requests: {status['requests']['active']}/{status['requests']['max_concurrent']}")
                print(f"   Uptime: {status['uptime_seconds']}s")
            else:
                print(f"[ERROR] API error: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] API not responding: {e}")
    
    def wait_for_server_ready(self, timeout: int = 180) -> bool:
        """Wait for server to be ready"""
        import requests

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://localhost:8080/health", timeout=5)
                if response.status_code == 200:
                    health = response.json()
                    status = health.get('status')
                    if status == 'healthy':
                        return True
                    elif status == 'loading':
                        print("L", end="", flush=True)  # Show 'L' for loading
                    else:
                        print(".", end="", flush=True)  # Show '.' for other states
                else:
                    print(".", end="", flush=True)
            except Exception:
                print(".", end="", flush=True)

            time.sleep(2)

        return False
    
    # Utility methods
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print application header"""
        print("=" * 60)
        print("         LLM SERVER MANAGEMENT SYSTEM")
        print("=" * 60)
        print()

def main():
    """Main entry point"""
    manager = LLMManager()
    
    # Check Docker availability
    if not manager.check_docker_available():
        print("[ERROR] Docker not found!")
        print("Please install Docker Desktop and make sure it's running.")
        input("Press Enter to exit...")
        return
    
    manager.main_menu()

if __name__ == "__main__":
    main()