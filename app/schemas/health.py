from .base import CamelModel

class HealthResponse(CamelModel):
    """Response model for health check."""
    status: str
    message: str
