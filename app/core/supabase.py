import logging
from typing import Any
from supabase import create_client, Client
from app.core.config import settings

logger = logging.getLogger(__name__)


class _UnconfiguredSupabaseClient:
    """Placeholder client that warns on startup and errors only when invoked."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item: str) -> Any:
        raise RuntimeError(
            f"{self._name} is not configured! Please set SUPABASE_URL and "
            f"SUPABASE_ANON_KEY / SUPABASE_SERVICE_ROLE_KEY in your environment variables."
        )


if settings.supabase_url and settings.supabase_anon_key:
    try:
        supabase: Client = create_client(settings.supabase_url, settings.supabase_anon_key)
    except Exception as exc:
        logger.error(f"Failed to initialize Supabase client: {exc}")
        supabase = _UnconfiguredSupabaseClient("supabase")  # type: ignore[assignment]
else:
    logger.warning("SUPABASE_URL or SUPABASE_ANON_KEY not set; Supabase client is unconfigured.")
    supabase = _UnconfiguredSupabaseClient("supabase")  # type: ignore[assignment]

if settings.supabase_url and settings.supabase_service_role_key:
    try:
        supabase_admin: Client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    except Exception as exc:
        logger.error(f"Failed to initialize Supabase admin client: {exc}")
        supabase_admin = _UnconfiguredSupabaseClient("supabase_admin")  # type: ignore[assignment]
else:
    logger.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set; Supabase admin client is unconfigured.")
    supabase_admin = _UnconfiguredSupabaseClient("supabase_admin")  # type: ignore[assignment]
