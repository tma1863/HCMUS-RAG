"""
Health Check API for HippoRAG Docker deployment
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
from datetime import datetime
import sys

# Import your modules - FIXED IMPORTS
try:
    from config.docker_config import DockerConfig
    from utils.ollama_client import OllamaClient
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HippoRAG Health API", version="1.0.0")

# Global variables
config = None
ollama_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize configuration and clients on startup"""
    global config, ollama_client
    
    try:
        config = DockerConfig()
        logger.info("Configuration loaded successfully")
        
        # Initialize Ollama client
        ollama_url = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        model_name = os.getenv("LLM_NAME", "llama3:8b")
        
        ollama_client = OllamaClient(base_url=ollama_url, model=model_name)
        logger.info(f"Ollama client initialized with model: {model_name}")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if configuration is loaded
        if config is None:
            raise HTTPException(status_code=503, detail="Configuration not loaded")
        
        # Check Ollama connection
        ollama_status = "disconnected"
        if ollama_client:
            try:
                ollama_status = "connected" if ollama_client.health_check() else "disconnected"
            except Exception as e:
                logger.warning(f"Ollama health check failed: {e}")
                ollama_status = "error"
        
        # Check if required directories exist
        required_dirs = ["/app/outputs", "/app/logs", "/app/embedding_stores"]
        missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
        
        if missing_dirs:
            logger.warning(f"Missing directories: {missing_dirs}")
        
        return JSONResponse(content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "config": "loaded",
                "ollama": ollama_status,
                "directories": "ok" if not missing_dirs else f"missing: {missing_dirs}",
                "storage": "file-based (parquet + pickle)"
            },
            "model": {
                "llm": os.getenv("LLM_NAME", "llama3:8b"),
                "embedding": os.getenv("EMBEDDING_MODEL", "facebook/contriever"),
                "temperature": float(os.getenv("TEMPERATURE", "0.0"))
            }
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "api": "HippoRAG",
        "version": "1.0.0",
        "docker": True,
        "architecture": "file-based-storage",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/models")
async def list_models():
    """List available models"""
    try:
        if ollama_client:
            models = ollama_client.list_models()
            return {"models": models}
        else:
            raise HTTPException(status_code=503, detail="Ollama client not available")
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test-generation")
async def test_generation(prompt: dict = {"text": "Hello, how are you?"}):
    """Test text generation"""
    try:
        if not ollama_client:
            raise HTTPException(status_code=503, detail="Ollama client not available")
        
        input_prompt = prompt.get("text", "Hello, how are you?") if isinstance(prompt, dict) else str(prompt)
        
        response = ollama_client.generate(
            input_prompt, 
            temperature=float(os.getenv("TEMPERATURE", "0.0")), 
            max_new_tokens=100
        )
        
        return {
            "prompt": input_prompt,
            "response": response,
            "model": os.getenv("LLM_NAME", "llama3:8b"),
            "temperature": float(os.getenv("TEMPERATURE", "0.0")),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in test generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info") 