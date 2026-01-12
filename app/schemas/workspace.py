from typing import Optional, List
from datetime import datetime
from uuid import UUID
from pydantic import Field
from app.schemas.base import CamelModel


class WorkspaceBase(CamelModel):
    name: str
    description: Optional[str] = None
    file_ids: List[UUID] = Field(default_factory=list)
    folder_ids: List[UUID] = Field(default_factory=list)


class WorkspaceCreate(WorkspaceBase):
    """Schema for creating a workspace - company_id is derived from auth token"""
    pass


class WorkspaceResponse(WorkspaceBase):
    id: UUID
    company_id: str
    created_at: datetime
    created_by: UUID

    class Config:
        from_attributes = True


class WorkspaceDetailResponse(WorkspaceResponse):
    """Potential expansion: resolve file/folder details into objects"""
    pass
