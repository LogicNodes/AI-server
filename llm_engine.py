#!/usr/bin/env python3
"""
LLM Inference Engine
Clean, efficient model loading and inference
"""

import os
import gc
import logging
import torch
from typing import Dict, List, Optional, Generator
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMEngine:
    """
    Unified LLM inference engine
    - Supports both transformers and GGUF models
    - Auto-detects model type
    - Manages GPU memory
    - Handles text generation
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.engine_type = None  # 'transformers' or 'gguf'

        # Import transformers here to avoid loading time if not needed
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
        except ImportError:
            logger.error("transformers library not installed. Run: pip install transformers torch")
            raise

        # Import GGUF engine
        try:
            from gguf_engine import GGUFEngine
            self.gguf_engine = GGUFEngine()
        except ImportError:
            logger.warning("GGUF engine not available. Install llama-cpp-python for GGUF support.")
            self.gguf_engine = None
    
    def load_model(self, model_path: str) -> bool:
        """Load a model from local path (auto-detects GGUF vs transformers)"""
        model_path = Path(model_path)

        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return False

        # Check if model is already loaded
        if self.current_model == str(model_path):
            logger.info(f"Model {model_path.name} already loaded")
            return True

        # Unload current model if any
        if self.model is not None:
            self.unload_model()

        # Auto-detect model type
        gguf_files = list(model_path.glob("**/*.gguf")) if model_path.is_dir() else ([model_path] if model_path.suffix == '.gguf' else [])
        has_config_json = (model_path / "config.json").exists() if model_path.is_dir() else False

        if gguf_files and self.gguf_engine:
            # Load as GGUF model
            logger.info(f"Detected GGUF model, using llama.cpp engine")
            success = self.gguf_engine.load_model(str(model_path))
            if success:
                self.engine_type = 'gguf'
                self.current_model = str(model_path)
            return success

        elif has_config_json:
            # Load as transformers model
            logger.info(f"Detected transformers model, using HuggingFace engine")
            return self._load_transformers_model(model_path)

        else:
            logger.error(f"Unknown model format in {model_path}")
            return False

    def _load_transformers_model(self, model_path: Path) -> bool:
        """Load model using transformers library"""
        try:
            logger.info(f"Loading transformers model from {model_path}")

            # Load tokenizer
            self.tokenizer = self.AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with optimal settings
            self.model = self.AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True
            )

            # Move to device if not using device_map
            if self.device == "cuda" and not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.device)

            # Set to eval mode
            self.model.eval()

            self.engine_type = 'transformers'
            self.current_model = str(model_path)
            logger.info(f"Successfully loaded transformers model {model_path.name}")

            # Log memory usage
            if self.device == "cuda":
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"GPU memory used: {memory_used:.2f}GB")

            return True

        except Exception as e:
            logger.error(f"Failed to load transformers model {model_path}: {e}")
            self.unload_model()
            return False
    
    def unload_model(self):
        """Unload current model and free memory"""
        if self.engine_type == 'gguf' and self.gguf_engine:
            self.gguf_engine.unload_model()
        else:
            if self.model is not None:
                logger.info("Unloading transformers model")
                del self.model
                self.model = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

        self.current_model = None
        self.engine_type = None

        # Force garbage collection
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        if self.engine_type == 'gguf' and self.gguf_engine:
            return self.gguf_engine.is_model_loaded()
        else:
            return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about currently loaded model"""
        if not self.is_model_loaded():
            return {"status": "no_model_loaded"}

        if self.engine_type == 'gguf' and self.gguf_engine:
            return self.gguf_engine.get_model_info()
        else:
            info = {
                "status": "loaded",
                "model_path": self.current_model,
                "device": self.device,
                "model_type": "transformers"
            }

            if self.device == "cuda":
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                info["gpu_memory_used"] = f"{memory_used:.2f}GB"

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

        if self.engine_type == 'gguf' and self.gguf_engine:
            return self.gguf_engine.generate_response(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                stream=stream
            )
        else:
            return self._generate_transformers_response(
                messages, max_tokens, temperature, do_sample, stream
            )

    def _generate_transformers_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        do_sample: bool,
        stream: bool
    ) -> str:
        """Generate response using transformers"""
        try:
            # Format messages into prompt
            prompt = self._format_messages(messages)

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048  # Reasonable context limit
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            # Generate
            with torch.no_grad():
                if stream:
                    return self._generate_stream(inputs, generation_kwargs)
                else:
                    outputs = self.model.generate(**inputs, **generation_kwargs)

                    # Decode response
                    input_length = inputs["input_ids"].shape[1]
                    generated_tokens = outputs[0][input_length:]
                    response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                    return response.strip()

        except Exception as e:
            logger.error(f"Transformers generation failed: {e}")
            raise
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string"""
        # Simple chat format - can be enhanced for specific models
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
    
    def _generate_stream(self, inputs, generation_kwargs) -> Generator[str, None, None]:
        """Generate streaming response (placeholder for future implementation)"""
        # For now, just return the full response
        # TODO: Implement proper streaming with text_streamer
        outputs = self.model.generate(**inputs, **generation_kwargs)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        yield response.strip()
    
    def calculate_tokens(self, text: str) -> int:
        """Calculate token count for text"""
        if self.engine_type == 'gguf' and self.gguf_engine:
            return self.gguf_engine.calculate_tokens(text)
        elif self.tokenizer:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        else:
            return len(text.split())  # Rough estimate

# Global engine instance
_engine = None

def get_engine() -> LLMEngine:
    """Get global LLM engine instance"""
    global _engine
    if _engine is None:
        _engine = LLMEngine()
    return _engine

def test_engine():
    """Test the LLM engine"""
    engine = get_engine()
    
    # Test model loading
    models_dir = Path("models")
    if models_dir.exists():
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if model_dirs:
            test_model = model_dirs[0]
            print(f"Testing with model: {test_model.name}")
            
            success = engine.load_model(str(test_model))
            if success:
                print("✓ Model loaded successfully")
                
                # Test generation
                messages = [{"role": "user", "content": "Hello! How are you?"}]
                response = engine.generate_response(messages, max_tokens=50)
                print(f"✓ Generated response: {response}")
                
                # Test info
                info = engine.get_model_info()
                print(f"✓ Model info: {info}")
                
            else:
                print("✗ Failed to load model")
        else:
            print("No models found in models/ directory")
    else:
        print("models/ directory not found")

if __name__ == "__main__":
    test_engine()