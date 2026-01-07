from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # Ollama Configuration (Local Embeddings)
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    
    # OpenRouter Configuration (LLM)
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Optional: Direct OpenAI key for alternative embeddings
    openai_api_key: Optional[str] = None
    
    # Chunking Configuration
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    # Vector Database
    chroma_persist_directory: str = "./chroma_db"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    
    # Supabase Configuration
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    
    # AWS S3 Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "eu-north-1"
    s3_bucket_name: str = ""
    s3_base_url: Optional[str] = None  # Optional: Custom S3 base URL (e.g., CloudFront)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()
