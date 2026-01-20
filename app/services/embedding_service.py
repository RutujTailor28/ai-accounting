from typing import List, Optional
import requests
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from app.core.config import settings


class EmbeddingService:
    """Service for generating embeddings with Ollama (Local) and FastEmbed (Fallback)."""

    def __init__(self):
        """Initialize embedding service, prioritizing Ollama with FastEmbed fallback."""
        self.use_fallback = False

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
            print(f"[WARNING] Could not connect to Ollama: {e}. Falling back to FastEmbed embeddings.")
            self.use_fallback = True
            try:
                # FastEmbed is lightweight and doesn't require torch
                self.embeddings = FastEmbedEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5"
                )
                print(f"[INFO] EmbeddingService initialized with FastEmbed (BGE-Small) fallback")
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