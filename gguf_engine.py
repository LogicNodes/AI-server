#!/usr/bin/env python3
"""
GGUF Model Engine using llama.cpp
Optimized for quantized models (Q4, Q5, Q8)
"""

import os
import gc
import logging
from typing import Dict, List, Optional
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GGUFEngine:
    """
    GGUF inference engine using llama.cpp
    - Optimized for quantized models
    - Better performance than transformers for GGUF
    - Lower memory usage
    """

    def __init__(self):
        self.model = None
        self.current_model = None
        self.model_params = {}

        # Import llama-cpp-python here
        try:
            from llama_cpp import Llama
            self.Llama = Llama
        except ImportError:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python[server]")
            raise

    def load_model(self, model_path: str, **kwargs) -> bool:
        """Load a GGUF model from local path"""
        model_path = Path(model_path)

        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return False

        # Find GGUF file in directory or use direct path
        if model_path.is_dir():
            gguf_files = list(model_path.glob("*.gguf"))
            if not gguf_files:
                logger.error(f"No GGUF files found in {model_path}")
                return False

            # Prefer Q4_K_M, then Q5_K_M, then others
            preferred_order = ["q4_k_m", "q5_k_m", "q4_0", "q8_0", "f16"]
            gguf_file = None

            for pref in preferred_order:
                for f in gguf_files:
                    if pref in f.name.lower():
                        gguf_file = f
                        break
                if gguf_file:
                    break

            if not gguf_file:
                gguf_file = gguf_files[0]  # Use first available
        else:
            gguf_file = model_path

        # Check if model is already loaded
        if self.current_model == str(gguf_file):
            logger.info(f"Model {gguf_file.name} already loaded")
            return True

        # Unload current model if any
        if self.model is not None:
            self.unload_model()

        try:
            logger.info(f"Loading GGUF model from {gguf_file}")

            # Load model with optimal settings
            # Calculate optimal GPU layers based on available VRAM
            gpu_layers = self._calculate_optimal_gpu_layers(gguf_file)

            self.model = self.Llama(
                model_path=str(gguf_file),
                n_ctx=8192,  # Context window
                n_batch=512,  # Batch size
                n_gpu_layers=gpu_layers,  # Optimized GPU/RAM split
                verbose=False,
                **kwargs
            )

            self.current_model = str(gguf_file)
            self.model_params = {
                'path': str(gguf_file),
                'name': gguf_file.stem,
                'size_mb': gguf_file.stat().st_size / (1024*1024)
            }

            logger.info(f"Successfully loaded GGUF model {gguf_file.name}")
            logger.info(f"Model size: {self.model_params['size_mb']:.1f}MB")

            return True

        except Exception as e:
            logger.error(f"Failed to load GGUF model {gguf_file}: {e}")
            self.unload_model()
            return False

    def unload_model(self):
        """Unload current model and free memory"""
        if self.model is not None:
            logger.info("Unloading GGUF model")
            del self.model
            self.model = None

        self.current_model = None
        self.model_params = {}

        # Force garbage collection
        gc.collect()

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self.model is not None

    def get_model_info(self) -> Dict[str, str]:
        """Get information about currently loaded model"""
        if not self.is_model_loaded():
            return {"status": "no_model_loaded"}

        info = {
            "status": "loaded",
            "model_path": self.current_model,
            "model_type": "gguf",
            "size_mb": f"{self.model_params.get('size_mb', 0):.1f}MB"
        }

        return info

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        stream: bool = False
    ) -> str:
        """Generate response from messages"""

        if not self.is_model_loaded():
            raise RuntimeError("No model loaded")

        try:
            # Format messages into prompt
            prompt = self._format_messages(messages)

            # Generate response
            if stream:
                return self._generate_stream(prompt, max_tokens, temperature)
            else:
                output = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["User:", "Human:", "\n\n"],
                    echo=False
                )

                response = output['choices'][0]['text'].strip()
                return response

        except Exception as e:
            logger.error(f"GGUF generation failed: {e}")
            raise

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string"""
        formatted = ""

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                formatted += f"System: {content}\n"
            elif role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"

        # Add assistant prompt for response
        formatted += "Assistant:"

        return formatted

    def _generate_stream(self, prompt: str, max_tokens: int, temperature: float):
        """Generate streaming response"""
        stream = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["User:", "Human:", "\n\n"],
            echo=False,
            stream=True
        )

        full_response = ""
        for output in stream:
            token = output['choices'][0]['text']
            full_response += token

        return full_response.strip()

    def _calculate_optimal_gpu_layers(self, gguf_file: Path) -> int:
        """Calculate optimal number of GPU layers based on available VRAM and RAM"""
        try:
            import subprocess
            import psutil

            # Get available GPU memory
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                free_vram_mb = float(result.stdout.strip())
                free_vram_gb = free_vram_mb / 1024
            else:
                free_vram_gb = 3.5  # Conservative fallback

            # Get available system RAM
            memory = psutil.virtual_memory()
            free_ram_gb = memory.available / (1024**3)
            total_ram_gb = memory.total / (1024**3)

        except Exception as e:
            logger.warning(f"Memory detection failed: {e}")
            free_vram_gb = 3.5  # Conservative fallback
            free_ram_gb = 8.0   # Conservative fallback
            total_ram_gb = 16.0

        # Get model size
        model_size_gb = gguf_file.stat().st_size / (1024**3)

        logger.info(f"System Memory - Model: {model_size_gb:.1f}GB, VRAM: {free_vram_gb:.1f}GB, RAM: {free_ram_gb:.1f}GB/{total_ram_gb:.1f}GB")

        # Check if model fits entirely on GPU
        if model_size_gb <= free_vram_gb * 0.9:  # 90% safety margin
            gpu_layers = -1  # All layers
            logger.info("[OK] Using all GPU layers (model fits in VRAM)")
            return gpu_layers

        # Calculate RAM requirements for hybrid mode
        ram_needed_gb = model_size_gb - (free_vram_gb * 0.8)  # 80% VRAM safety margin

        if ram_needed_gb > free_ram_gb * 0.7:  # 70% RAM safety margin
            # Not enough RAM for hybrid mode
            logger.warning("[WARNING] Insufficient RAM for hybrid mode")
            logger.warning(f"   Need: {ram_needed_gb:.1f}GB RAM, Available: {free_ram_gb:.1f}GB")
            logger.warning(f"   Model too large - will run very slowly or fail")

            # Try minimal GPU layers (emergency mode)
            gpu_layers = max(1, int(free_vram_gb / model_size_gb * 4))  # Very conservative
            logger.info(f"[EMERGENCY] {gpu_layers} GPU layers, {model_size_gb - 1:.1f}GB in RAM")
            return gpu_layers

        # Calculate optimal hybrid split
        gpu_ratio = (free_vram_gb * 0.8) / model_size_gb
        estimated_layers = int(32 * gpu_ratio)  # Assume ~32 layers typical
        gpu_layers = max(1, min(estimated_layers, 31))

        logger.info(f"[HYBRID] {gpu_layers} GPU layers, {ram_needed_gb:.1f}GB in RAM")
        logger.info(f"   Expected performance: {int(gpu_ratio * 100)}% of full GPU speed")

        return gpu_layers

    def calculate_tokens(self, text: str) -> int:
        """Calculate token count for text"""
        if not self.model:
            return len(text.split())  # Rough estimate

        # llama.cpp tokenization
        tokens = self.model.tokenize(text.encode('utf-8'))
        return len(tokens)

def test_gguf_engine():
    """Test the GGUF engine"""
    engine = GGUFEngine()

    # Test model loading
    models_dir = Path("models")
    if models_dir.exists():
        # Look for GGUF files
        gguf_files = list(models_dir.glob("**/*.gguf"))
        if gguf_files:
            test_model = gguf_files[0].parent
            print(f"Testing with GGUF model: {test_model.name}")

            success = engine.load_model(str(test_model))
            if success:
                print("✓ GGUF model loaded successfully")

                # Test generation
                messages = [{"role": "user", "content": "Hello! How are you?"}]
                response = engine.generate_response(messages, max_tokens=50)
                print(f"✓ Generated response: {response}")

                # Test info
                info = engine.get_model_info()
                print(f"✓ Model info: {info}")

            else:
                print("✗ Failed to load GGUF model")
        else:
            print("No GGUF files found in models/ directory")
    else:
        print("models/ directory not found")

if __name__ == "__main__":
    test_gguf_engine()