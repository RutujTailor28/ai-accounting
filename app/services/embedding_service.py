from typing import List, Optional
import requests
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings


class EmbeddingService:
    """Service for generating embeddings with Ollama (Local) and HuggingFace (Fallback)."""
    
    def __init__(self):
        """Initialize embedding service, trying Ollama first then HuggingFace."""
        self.use_fallback = False
        
        # Try to connect to Ollama
        try:
            print(f"[INFO] Attempting to connect to Ollama at {settings.ollama_base_url}...")
            # Simple heartbeat check
            response = requests.get(settings.ollama_base_url, timeout=2)
            if response.status_code == 200:
                self.embeddings = OllamaEmbeddings(
                    model=settings.embedding_model,
                    base_url=settings.ollama_base_url
                )
                print(f"[INFO] EmbeddingService initialized with Ollama model={settings.embedding_model}")
            else:
                raise ConnectionError(f"Ollama returned status {response.status_code}")
        except Exception as e:
            print(f"[WARNING] Could not connect to Ollama: {e}. Falling back to HuggingFace embeddings.")
            self.use_fallback = True
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=settings.huggingface_model
                )
                print(f"[INFO] EmbeddingService initialized with HuggingFace model={settings.huggingface_model} (FREE/LOCAL)")
            except Exception as hf_e:
                print(f"[ERROR] Failed to initialize HuggingFace embeddings: {hf_e}")
                raise hf_e
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        """
        try:
            embedding = self.embeddings.embed_query(text)
            # print(f"[DEBUG] Generated embedding of dimension {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"[ERROR] Error generating embedding: {str(e)}")
            # If Ollama fails mid-way, we might want to retry with fallback if not already using it
            if not self.use_fallback:
                 print("[INFO] Mid-execution Ollama failure. Re-initializing with HuggingFace...")
                 self.use_fallback = True
                 self.embeddings = HuggingFaceEmbeddings(model_name=settings.huggingface_model)
                 return self.embeddings.embed_query(text)
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            print(f"[INFO] Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            print(f"[ERROR] Error generating embeddings: {str(e)}")
            if not self.use_fallback:
                 print("[INFO] Mid-execution Ollama failure. Re-initializing with HuggingFace...")
                 self.use_fallback = True
                 self.embeddings = HuggingFaceEmbeddings(model_name=settings.huggingface_model)
                 return self.embeddings.embed_documents(texts)
            raise
