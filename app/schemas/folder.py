from typing import Optional, List
from datetime import datetime
from uuid import UUID
from pydantic import Field
from app.schemas.base import CamelModel


class FolderBase(CamelModel):
    name: str
    company_id: str
    parent_id: Optional[UUID] = None


class FolderCreate(FolderBase):
    pass


class FolderResponse(FolderBase):
    id: UUID
    created_at: datetime
    created_by: UUID

    class Config:
        from_attributes = True


class FileBase(CamelModel):
    name: str
    folder_id: UUID
    s3_key: str
    s3_url: str
    file_type: str
    company_id: str


class FileResponse(FileBase):
    id: UUID
    uploaded_by: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class FolderWithFilesResponse(FolderResponse):
    files: List[FileResponse] = Field(default_factory=list)
