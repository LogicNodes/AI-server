#!/usr/bin/env python3
"""
Docker-optimized LLM Server
Designed to run in containers with mounted model volumes
"""

import os
import logging
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from llm_engine import get_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "/models")
API_KEY = os.getenv("API_KEY", "hardq_dev_key_001")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "2"))
MAX_TOKENS_LIMIT = int(os.getenv("MAX_TOKENS_LIMIT", "1000"))
PORT = int(os.getenv("PORT", "8080"))

# Auto-detect model if MODEL_PATH is a directory with single model
if Path(MODEL_PATH).is_dir():
    model_dirs = [d for d in Path(MODEL_PATH).iterdir() if d.is_dir() and not d.name.startswith('.')]
    if len(model_dirs) == 1:
        MODEL_PATH = str(model_dirs[0])
        logger.info(f"Auto-detected model: {MODEL_PATH}")
    # If MODEL_PATH points to a model directory directly, check if it has config.json
    elif Path(MODEL_PATH, "config.json").exists():
        logger.info(f"Using direct model path: {MODEL_PATH}")
    else:
        logger.warning(f"No valid model found in {MODEL_PATH}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Docker LLM Server")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"API key: {API_KEY}")
    logger.info(f"Max concurrent: {MAX_CONCURRENT}")

    # Verify model path exists
    if not Path(MODEL_PATH).exists():
        logger.error(f"Model path does not exist: {MODEL_PATH}")
        logger.error("Make sure the model volume is properly mounted")
    else:
        # Load model
        global model_loading
        engine = get_engine()
        logger.info("Loading model... this may take a few minutes")
        model_loading = True
        success = engine.load_model(MODEL_PATH)
        model_loading = False

        if success:
            logger.info("✅ Model loaded successfully")
            gpu_info = get_gpu_info()
            if "error" not in gpu_info:
                logger.info(f"GPU Memory: {gpu_info['memory_used_gb']:.1f}GB used, {gpu_info['memory_free_gb']:.1f}GB free")
        else:
            logger.error("❌ Model loading failed")

    yield

    # Shutdown
    logger.info("Shutting down server")
    engine = get_engine()
    engine.unload_model()

app = FastAPI(
    title="Docker LLM Server",
    description="Containerized LLM server with mounted model volumes",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_requests = 0
total_requests = 0
start_time = datetime.utcnow()
model_loading = False

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict[str, int]

def get_gpu_info():
    """Get GPU information"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            used_mb, total_mb, util_percent, temp = result.stdout.strip().split(', ')
            return {
                "memory_used_gb": float(used_mb) / 1024,
                "memory_total_gb": float(total_mb) / 1024,
                "memory_free_gb": (float(total_mb) - float(used_mb)) / 1024,
                "utilization_percent": float(util_percent),
                "temperature_c": float(temp)
            }
    except Exception as e:
        logger.warning(f"GPU info failed: {e}")
    
    return {"error": "GPU info not available"}

def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )
    
    token = authorization.split(" ", 1)[1]
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return token

def check_capacity():
    """Check if we can handle more requests"""
    global active_requests
    
    if active_requests >= MAX_CONCURRENT:
        return False, f"Server busy: {active_requests}/{MAX_CONCURRENT} requests active"
    
    engine = get_engine()
    if not engine.is_model_loaded():
        return False, "Model not loaded"
    
    # Check GPU memory
    gpu_info = get_gpu_info()
    if "error" not in gpu_info and gpu_info["memory_free_gb"] < 0.3:
        return False, f"Low GPU memory: {gpu_info['memory_free_gb']:.1f}GB free"
    
    return True, "Ready"


@app.get("/")
async def root():
    """Server status"""
    engine = get_engine()
    gpu_info = get_gpu_info()
    
    return {
        "status": "running",
        "container": "docker",
        "model": {
            "path": MODEL_PATH,
            "name": Path(MODEL_PATH).name,
            "loaded": engine.is_model_loaded()
        },
        "gpu": gpu_info,
        "requests": {
            "active": active_requests,
            "max_concurrent": MAX_CONCURRENT,
            "total_processed": total_requests
        },
        "uptime_seconds": int((datetime.utcnow() - start_time).total_seconds()),
        "version": "2.0.0-docker"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    global model_loading
    engine = get_engine()

    if model_loading:
        status = "loading"
        message = "Model is loading"
    else:
        can_serve, message = check_capacity()
        status = "healthy" if can_serve else "busy"

    return {
        "status": status,
        "model_loaded": engine.is_model_loaded(),
        "model_loading": model_loading,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    """List available models"""
    verify_api_key(authorization)
    
    engine = get_engine()
    if not engine.is_model_loaded():
        return {"object": "list", "data": []}
    
    model_name = Path(MODEL_PATH).name
    
    return {
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": int(start_time.timestamp()),
            "owned_by": "local"
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: str = Header(None)
):
    """Chat completions endpoint"""
    global active_requests, total_requests
    
    # Verify API key
    verify_api_key(authorization)
    
    # Check capacity
    can_serve, reason = check_capacity()
    if not can_serve:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=reason
        )
    
    # Validate request
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Messages cannot be empty"
        )
    
    if request.max_tokens and request.max_tokens > MAX_TOKENS_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"max_tokens cannot exceed {MAX_TOKENS_LIMIT}"
        )
    
    # Use loaded model if not specified, otherwise validate
    model_name = Path(MODEL_PATH).name
    if request.model is None:
        request.model = model_name
    elif request.model != model_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{request.model}' not available. Container model: '{model_name}'"
        )
    
    # Process request
    active_requests += 1
    total_requests += 1
    request_id = f"docker-{int(time.time())}-{total_requests}"
    
    try:
        logger.info(f"Processing {request_id}: {len(request.messages)} messages")
        start_time_req = time.time()
        
        # Generate response
        engine = get_engine()
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        response_text = engine.generate_response(
            messages=messages,
            max_tokens=request.max_tokens or 100,
            temperature=request.temperature or 0.7
        )
        
        # Calculate tokens
        prompt_text = " ".join([msg["content"] for msg in messages])
        prompt_tokens = engine.calculate_tokens(prompt_text)
        completion_tokens = engine.calculate_tokens(response_text)
        
        processing_time = time.time() - start_time_req
        logger.info(f"Completed {request_id}: {completion_tokens} tokens in {processing_time:.1f}s")
        
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=model_name,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )
    
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )
    
    finally:
        active_requests -= 1

if __name__ == "__main__":
    # Validate environment
    if not Path(MODEL_PATH).exists():
        logger.error(f"Model path {MODEL_PATH} not found!")
        logger.error("Make sure to mount your models directory to /models")
        exit(1)
    
    logger.info(f"Starting server on 0.0.0.0:{PORT}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )