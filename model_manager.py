#!/usr/bin/env python3
"""
Model Management System
- Download models from Hugging Face
- Auto-detect model specifications
- Manage local model repository
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def list_available_models(self) -> List[Dict[str, str]]:
        """List all locally available models with their specs"""
        models = []
        
        for model_path in self.models_dir.iterdir():
            if model_path.is_dir():
                specs = self.detect_model_specs(model_path.name)
                models.append({
                    'name': model_path.name,
                    'path': str(model_path),
                    'size_gb': specs['size_gb'],
                    'vram_gb': specs['vram_gb'],
                    'params': specs['params'],
                    'max_context': specs['max_context']
                })
        
        return sorted(models, key=lambda x: x['size_gb'])
    
    def detect_model_specs(self, model_name: str) -> Dict[str, str]:
        """Auto-detect model specifications from config and folder contents"""
        model_path = self.models_dir / model_name
        
        # Try to read config.json for model info
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Extract key parameters
                vocab_size = config.get('vocab_size', 50000)
                hidden_size = config.get('hidden_size', 2048)
                num_layers = config.get('num_hidden_layers', 24)
                max_position_embeddings = config.get('max_position_embeddings', 8192)
                
                # Estimate parameters (rough calculation)
                # Embedding: vocab_size * hidden_size
                # Transformer layers: num_layers * (hidden_size^2 * 4 + other weights)
                embedding_params = vocab_size * hidden_size
                layer_params = num_layers * (hidden_size * hidden_size * 12)  # Rough estimate
                total_params = embedding_params + layer_params
                
                # Convert to billions
                params_b = total_params / 1e9
                
                # Estimate VRAM needed (params * 2 bytes for FP16 + overhead)
                vram_gb = params_b * 2.0 * 1.2  # 20% overhead
                
            except Exception as e:
                # Fallback to name-based detection
                return self._detect_from_name(model_name)
        else:
            return self._detect_from_name(model_name)
        
        # Calculate disk size
        size_gb = self._calculate_folder_size(model_path)
        
        return {
            'params': f"{params_b:.1f}B",
            'vram_gb': f"{vram_gb:.1f}",
            'size_gb': f"{size_gb:.1f}",
            'max_context': str(max_position_embeddings)
        }
    
    def _detect_from_name(self, model_name: str) -> Dict[str, str]:
        """Fallback detection based on model name patterns"""
        name_lower = model_name.lower()
        
        if '1.5b' in name_lower or '1_5b' in name_lower:
            return {'params': '1.5B', 'vram_gb': '3.0', 'size_gb': '3.0', 'max_context': '8192'}
        elif '7b' in name_lower:
            return {'params': '7B', 'vram_gb': '14.0', 'size_gb': '14.0', 'max_context': '8192'}
        elif '3b' in name_lower:
            return {'params': '3B', 'vram_gb': '6.0', 'size_gb': '6.0', 'max_context': '8192'}
        elif '13b' in name_lower:
            return {'params': '13B', 'vram_gb': '26.0', 'size_gb': '26.0', 'max_context': '4096'}
        else:
            return {'params': 'Unknown', 'vram_gb': '4.0', 'size_gb': '4.0', 'max_context': '2048'}
    
    def _calculate_folder_size(self, folder_path: Path) -> float:
        """Calculate folder size in GB"""
        total_size = 0
        try:
            for file_path in folder_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            return 0.0
        
        return total_size / (1024**3)  # Convert to GB
    
    def download_model(self, huggingface_url: str, quantization: Optional[str] = None) -> Tuple[bool, str]:
        """Download a model from Hugging Face"""
        try:
            # Parse the URL to get model name
            if 'huggingface.co' not in huggingface_url:
                return False, "Invalid Hugging Face URL"
            
            # Extract model name from URL (e.g., microsoft/DialoGPT-small)
            parts = huggingface_url.split('/')
            if len(parts) < 2:
                return False, "Could not parse model name from URL"
            
            model_name = f"{parts[-2]}--{parts[-1]}"  # org--model format for folder
            model_path = self.models_dir / model_name
            
            if model_path.exists():
                return False, f"Model {model_name} already exists"
            
            print(f"Downloading {model_name}...")
            print("This may take a while depending on model size...")
            
            # Use huggingface-hub to download
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'huggingface-hub', '--quiet'
            ], capture_output=True)
            
            if result.returncode != 0:
                return False, "Failed to install huggingface-hub"
            
            # Download using huggingface-hub
            download_script = f"""
import sys
from huggingface_hub import snapshot_download

try:
    model_id = "{parts[-2]}/{parts[-1]}"
    local_dir = "{str(model_path).replace(chr(92), '/')}"
    
    print(f"Downloading {{model_id}} to {{local_dir}}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print("Download completed successfully!")
    
except Exception as e:
    print(f"Download failed: {{e}}")
    sys.exit(1)
"""
            
            with open("temp_download.py", 'w') as f:
                f.write(download_script)
            
            result = subprocess.run([sys.executable, "temp_download.py"], 
                                  capture_output=False, text=True)
            
            # Clean up
            if os.path.exists("temp_download.py"):
                os.remove("temp_download.py")
            
            if result.returncode == 0 and model_path.exists():
                # Verify download
                specs = self.detect_model_specs(model_name)
                return True, f"Successfully downloaded {model_name} ({specs['params']} parameters, ~{specs['vram_gb']}GB VRAM)"
            else:
                return False, "Download failed or incomplete"
                
        except Exception as e:
            return False, f"Download error: {str(e)}"
    
    def validate_gpu_compatibility(self, model_name: str) -> Tuple[bool, str, Dict[str, float]]:
        """Check if model fits on available GPU"""
        specs = self.detect_model_specs(model_name)
        model_vram = float(specs['vram_gb'])
        
        # Get current GPU memory
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.total,memory.used',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                total_mb, used_mb = result.stdout.strip().split(', ')
                total_gb = float(total_mb) / 1024
                used_gb = float(used_mb) / 1024
                free_gb = total_gb - used_gb
            else:
                return False, "Could not query GPU memory", {}
                
        except Exception as e:
            return False, f"GPU query failed: {e}", {}
        
        # Check if model fits with some overhead
        if model_vram > free_gb:
            return False, f"Insufficient VRAM: need {model_vram:.1f}GB, have {free_gb:.1f}GB free", {
                'total_gb': total_gb, 'used_gb': used_gb, 'free_gb': free_gb, 'model_vram': model_vram
            }
        
        # Calculate remaining memory for inference
        remaining_gb = free_gb - model_vram
        max_context_estimate = int(remaining_gb * 1000 / 0.5) if remaining_gb > 0 else 0  # Rough estimate
        
        return True, f"Compatible: {remaining_gb:.1f}GB available for inference (~{max_context_estimate} tokens)", {
            'total_gb': total_gb, 'used_gb': used_gb, 'free_gb': free_gb, 
            'model_vram': model_vram, 'inference_gb': remaining_gb
        }

def main():
    """Test the model manager"""
    manager = ModelManager()
    
    print("Available models:")
    models = manager.list_available_models()
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']}")
        print(f"   Parameters: {model['params']}")
        print(f"   VRAM needed: {model['vram_gb']}GB")
        print(f"   Disk size: {model['size_gb']}GB")
        print(f"   Max context: {model['max_context']}")
        
        # Test GPU compatibility
        compatible, msg, stats = manager.validate_gpu_compatibility(model['name'])
        print(f"   GPU compatible: {'OK' if compatible else 'NO'} {msg}")
        print()

if __name__ == "__main__":
    main()