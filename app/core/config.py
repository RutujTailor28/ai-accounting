from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application configuration settings."""

    # LLM Provider selection: 'nvidia', 'openrouter', 'openai', 'ollama'
    llm_provider: str = "nvidia"

    # NVIDIA NIM (build.nvidia.com)
    nvidia_api_key: str = ""
    nvidia_model: str = "meta/llama-3.1-8b-instruct"
    nvidia_model_accounting: str = "meta/llama-3.1-8b-instruct"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

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

    # ── Email / SMTP ──────────────────────────────────────
    # For Gmail: enable 2FA → create an App Password at myaccount.google.com/apppasswords
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 465
    smtp_user: str = ""          # e.g. hello@fineyukt.com
    smtp_password: str = ""      # Gmail App Password (16-char)
    admin_email: str = ""        # Where demo notifications go
    backend_base_url: str = "http://localhost:8000"
    frontend_base_url: str = "http://localhost:3000"

    @property
    def active_provider(self) -> str:
        """Resolve active LLM provider based on config and available API keys."""
        prov = (self.llm_provider or "").lower().strip()
        if prov == "nvidia" and self.nvidia_api_key:
            return "nvidia"
        if prov == "openrouter" and self.openrouter_api_key:
            return "openrouter"
        if self.nvidia_api_key:
            return "nvidia"
        if self.openrouter_api_key:
            return "openrouter"
        if self.openai_api_key:
            return "openai"
        return prov or "openrouter"

    @property
    def active_model(self) -> str:
        """Resolve the active model based on active provider."""
        if self.active_provider == "nvidia":
            return self.nvidia_model
        return self.openrouter_model

    @property
    def active_model_accounting(self) -> str:
        """Resolve the active accounting model based on active provider."""
        if self.active_provider == "nvidia":
            return self.nvidia_model_accounting
        return self.openrouter_model_accounting

    @property
    def active_api_key(self) -> str:
        """Resolve the active API key."""
        if self.active_provider == "nvidia":
            return self.nvidia_api_key
        if self.active_provider == "openai":
            return self.openai_api_key or ""
        return self.openrouter_api_key

    @property
    def active_base_url(self) -> str:
        """Resolve the active base URL."""
        if self.active_provider == "nvidia":
            return self.nvidia_base_url
        if self.active_provider == "openai":
            return "https://api.openai.com/v1"
        return self.openrouter_base_url

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

settings = Settings()
