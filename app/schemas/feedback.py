from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from app.schemas.base import CamelModel
from pydantic import Field

class FeedbackBase(CamelModel):
    original_query: str
    ai_response: str
    user_correction: str
    explanation: Optional[str] = None
    company_id: str
    customer_id: Optional[str] = None
    source_document: Optional[str] = None
    transaction_id: Optional[str] = None

class FeedbackCreate(FeedbackBase):
    pass

class FeedbackResponse(FeedbackBase):
    id: UUID
    created_at: datetime
    embedding_id: Optional[str] = None

    class Config:
        from_attributes = True

class FeedbackSearchRequest(CamelModel):
    query: str
    company_id: str
    limit: int = 3
