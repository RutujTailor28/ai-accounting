from app.services.embedding_service import EmbeddingService
from app.services.ai_service import LLMService

# Shared singleton instances to prevent redundant model loading and memory spikes
embedding_service = EmbeddingService()
llm_service = LLMService()
