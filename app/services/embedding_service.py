from typing import List, Optional
import requests
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from app.core.config import settings

class EmbeddingService:
    """Service for generating embeddings with Ollama (Local) and FastEmbed (Fallback)."""

    def __init__(self):
        """Initialize embedding service, forcing FastEmbed for deployment."""
        self.use_fallback = True

        try:

            self.embeddings = FastEmbedEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"
            )
            print(f"[INFO] EmbeddingService initialized with FastEmbed (BGE-Small) - FORCED FOR DEPLOY")
        except Exception as hf_e:
            print(f"[ERROR] Failed to initialize FastEmbed embeddings: {hf_e}")
            raise hf_e

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"[ERROR] Error generating embedding: {str(e)}")
            if not self.use_fallback:
                 print("[INFO] Mid-execution Ollama failure. Re-initializing with FastEmbed...")
                 self.use_fallback = True
                 self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
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
                 print("[INFO] Mid-execution Ollama failure. Re-initializing with FastEmbed...")
                 self.use_fallback = True
                 self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                 return self.embeddings.embed_documents(texts)
            raise
