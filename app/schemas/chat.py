from typing import Optional, List, Any
from datetime import datetime
from uuid import UUID
from pydantic import Field
from app.schemas.base import CamelModel


class ChatQueryRequest(CamelModel):
    workspace_id: UUID
    session_id: UUID
    question: str
    file_types: Optional[List[str]] = Field(None, description="Filter by file extensions (e.g. ['pdf', 'docx'])")
    folder_ids: Optional[List[str]] = Field(None, description="Filter by folder IDs")
    uploaded_by: Optional[List[str]] = Field(None, description="Filter by user IDs")
    start_date: Optional[str] = Field(None, description="Start date for filtering documents (ISO format)")
    end_date: Optional[str] = Field(None, description="End date for filtering documents (ISO format)")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")


class ChatMessageResponse(CamelModel):
    id: UUID
    session_id: UUID
    session_title: Optional[str] = None
    workspace_id: UUID
    role: str
    content: str
    data: Optional[Any] = None
    sources: Optional[List[str]] = Field(default_factory=list)
    created_at: datetime

    class Config:
        from_attributes = True


class ChatHistoryItem(CamelModel):
    session_id: UUID
    session_title: Optional[str] = None
    last_message_at: datetime
    workspace_id: UUID
