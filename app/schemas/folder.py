from typing import Optional, List
from datetime import datetime
from uuid import UUID
from pydantic import Field
from app.schemas.base import CamelModel

class FolderBase(CamelModel):
    name: str
    company_id: str
    parent_id: Optional[UUID] = None
    customer_id: Optional[UUID] = None

class FolderCreate(CamelModel):
    """Schema for creating a folder - company_id is derived from auth token"""
    name: str
    parent_id: Optional[UUID] = None
    customer_id: Optional[UUID] = None

class FolderResponse(FolderBase):
    id: UUID
    created_at: datetime
    created_by: UUID
    file_count: Optional[int] = 0
    total_size: Optional[int] = 0

    class Config:
        from_attributes = True

class FileBase(CamelModel):
    name: str
    folder_id: UUID
    s3_key: str
    s3_url: str
    file_type: str
    company_id: str
    size: Optional[int] = None

class FileResponse(FileBase):
    id: UUID
    created_by: UUID
    created_at: datetime

    class Config:
        from_attributes = True

class FolderWithFilesResponse(FolderResponse):
    files: List[FileResponse] = Field(default_factory=list)
