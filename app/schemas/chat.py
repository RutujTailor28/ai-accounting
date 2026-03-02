from typing import Optional, List, Any
from datetime import datetime
from uuid import UUID
from pydantic import Field
from app.schemas.base import CamelModel


class ChatQueryRequest(CamelModel):
    workspace_id: Optional[UUID] = Field(None, description="Filter by workspace ID")
    customer_id: Optional[UUID] = Field(None, description="Filter by customer ID")
    session_id: UUID
    question: str
    file_types: Optional[List[str]] = Field(None, description="Filter by file extensions (e.g. ['pdf', 'docx'])")
    file_ids: Optional[List[str]] = Field(None, description="Filter by exact file IDs")
    folder_ids: Optional[List[str]] = Field(None, description="Filter by folder IDs")
    uploaded_by: Optional[List[str]] = Field(None, description="Filter by user IDs")
    start_date: Optional[str] = Field(None, description="Start date for filtering documents (ISO format)")
    end_date: Optional[str] = Field(None, description="End date for filtering documents (ISO format)")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")


class ChatMessageResponse(CamelModel):
    id: UUID
    session_id: UUID
    session_title: Optional[str] = None
    workspace_id: Optional[UUID] = None
    customer_id: Optional[UUID] = None
    role: str
    content: str
    data: Optional[Any] = None
    sources: Optional[List[str]] = Field(default_factory=list)
    rating: Optional[int] = None
    created_at: datetime
    file_names: Optional[List[str]] = Field(default_factory=list)

    class Config:
        from_attributes = True


class ChatHistoryItem(CamelModel):
    session_id: UUID
    session_title: Optional[str] = None
    last_message_at: datetime
    workspace_id: Optional[UUID] = None
    customer_id: Optional[UUID] = None
    file_names: Optional[List[str]] = Field(default_factory=list)


class ChatSaveRequest(CamelModel):
    session_id: UUID
    file_names: List[str] = Field(default_factory=list)


class ChatFeedbackCreate(CamelModel):
    message_id: UUID
    rating: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = None
class ChatUpdateTitleRequest(CamelModel):
    session_title: str = Field(..., min_length=1, max_length=255)

class ChatMessageUpdateRequest(CamelModel):
    content: Optional[str] = None
    data: Optional[Any] = None
