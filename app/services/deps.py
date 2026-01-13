from app.services.embedding_service import EmbeddingService
from app.services.ai_service import LLMService

# Singleton instances of services
embedding_service = EmbeddingService()
llm_service = LLMService()
