from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application configuration settings."""

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    huggingface_model: str = "all-MiniLM-L6-v2"

    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    openrouter_model_accounting: str = "meta-llama/llama-3.3-70b-instruct:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    openai_api_key: Optional[str] = None

    chunk_size: int = 2000
    chunk_overlap: int = 200

    chroma_persist_directory: str = "./chroma_db"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    log_level: str = "INFO"

    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""

    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "eu-north-1"
    s3_bucket_name: str = ""
    s3_base_url: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

settings = Settings()
