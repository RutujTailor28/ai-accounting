from typing import List, Optional
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings


class EmbeddingService:
    """Service for generating embeddings using local Ollama."""
    
    def __init__(self):
        """Initialize embedding service with local Ollama model."""
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url
        )
        print(f"[INFO] EmbeddingService initialized with local model={settings.embedding_model} at {settings.ollama_base_url}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            print(f"[DEBUG] Generated embedding of dimension {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"[ERROR] Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            print(f"[INFO] Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            print(f"[ERROR] Error generating embeddings: {str(e)}")
            raise
