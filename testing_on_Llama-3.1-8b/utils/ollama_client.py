"""
Ollama API Client for HippoRAG
"""
import requests
import json
import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://ollama:11434", model: str = "llama3:8b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        
        logger.info(f"Initializing OllamaClient with base_url: {self.base_url}, model: {self.model}")
        
        # Test connection
        self._wait_for_ollama()
        self._ensure_model_available()
    
    def _wait_for_ollama(self, max_retries: int = 30, delay: int = 5):
        """Wait for Ollama service to be ready"""
        for i in range(max_retries):
            try:
                response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    logger.info("Successfully connected to Ollama")
                    return
                else:
                    logger.warning(f"Ollama responded with status {response.status_code}")
            except Exception as e:
                logger.warning(f"Waiting for Ollama... Attempt {i+1}/{max_retries}: {e}")
                time.sleep(delay)
        
        raise ConnectionError(f"Could not connect to Ollama at {self.base_url} after {max_retries} attempts")
    
    def _ensure_model_available(self):
        """Ensure the model is available, pull if necessary"""
        try:
            models = self.list_models()
            model_names = [m.get('name', '') for m in models]
            
            if not any(self.model in name for name in model_names):
                logger.info(f"Model {self.model} not found. Attempting to pull...")
                if self.pull_model():
                    logger.info(f"Successfully pulled model {self.model}")
                else:
                    logger.error(f"Failed to pull model {self.model}")
            else:
                logger.info(f"Model {self.model} is available")
                
        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama API"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.0),  # Default to 0.0
                    "num_predict": kwargs.get("max_new_tokens", 2048),
                    "top_p": kwargs.get("top_p", 0.9),
                    "stop": kwargs.get("stop", [])
                }
            }
            
            logger.debug(f"Sending request to {self.base_url}/api/generate")
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                return generated_text
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return ""
    
    def chat(self, messages: list, **kwargs) -> str:
        """Chat completion using Ollama API"""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.0),  # Default to 0.0
                    "num_predict": kwargs.get("max_new_tokens", 2048),
                    "top_p": kwargs.get("top_p", 0.9)
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                logger.error(f"Ollama chat API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling Ollama chat API: {e}")
            return ""
    
    def pull_model(self):
        """Pull/download the model if not already available"""
        try:
            payload = {"model": self.model}
            logger.info(f"Pulling model {self.model}...")
            
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=1800
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully pulled model {self.model}")
                return True
            else:
                logger.error(f"Failed to pull model: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    def list_models(self):
        """List available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=30)
            if response.status_code == 200:
                return response.json().get("models", [])
            else:
                logger.error(f"Failed to list models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def health_check(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False

    def test_generation(self) -> bool:
        """Test if generation works"""
        try:
            result = self.generate("Hello", temperature=0.0, max_new_tokens=10)
            success = len(result.strip()) > 0
            logger.info(f"Generation test: {'PASSED' if success else 'FAILED'}")
            return success
        except Exception as e:
            logger.error(f"Generation test failed: {e}")
            return False 