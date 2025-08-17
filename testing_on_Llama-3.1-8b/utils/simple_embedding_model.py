# simple_embedding_model.py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
import logging
from config.config import config  # Import config instance từ config.py

logger = logging.getLogger(__name__)

def mean_pooling(token_embeddings, mask):
    """Mean pooling function for Contriever-style embeddings"""
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

class SimpleEmbeddingModel:
    """Simplified embedding model cho HippoRAG architecture với Contriever support"""
    
    def __init__(self, model_name: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Sử dụng config từ config.py nếu không có model_name
        if model_name is None:
            model_name = config.EMBEDDING_MODEL or config.embedding_model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_name = model_name
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def batch_encode(self, texts: List[str], instruction: str = None, norm: bool = True) -> np.ndarray:
        """Encode texts to embeddings using appropriate pooling strategy"""
        if not texts:
            logger.warning("Empty texts list provided to batch_encode")
            return np.array([])
        
        logger.info(f"SIMPLIFIED DEBUG: Just essential info")
        logger.info(f"Encoding {len(texts)} texts")
        
        try:
            # Process in batches to avoid memory issues
            all_embeddings = []
            batch_size = 32
            
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch, 
                        padding=True, 
                        truncation=True, 
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Forward pass
                    outputs = self.model(**inputs)
                    
                    # Mean pooling
                    embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                    
                    # Normalize if requested
                    if norm:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    # Move to CPU and convert to numpy
                    batch_embeddings = embeddings.cpu().numpy()
                    all_embeddings.extend(batch_embeddings)
            
            # Convert to numpy array
            vecs = np.array(all_embeddings)
            
            # Validate output
            if vecs.shape[0] != len(texts):
                raise ValueError(f"Embedding count mismatch: {vecs.shape[0]} != {len(texts)}")
            
            logger.info(f"Encoded to shape {vecs.shape}")
            return vecs
            
        except Exception as e:
            logger.error(f"❌ Error encoding texts: {e}")
            # Only show first few texts on error for debugging
            if len(texts) <= 3:
                logger.error(f"📋 Failed texts: {texts}")
            else:
                logger.error(f"📋 Failed texts (first 3): {texts[:3]}...")
            raise
