from typing import Optional, List
from datetime import datetime
from uuid import UUID
from pydantic import Field
from app.schemas.base import CamelModel
from app.schemas.folder import FolderResponse

class CustomerBase(CamelModel):
    name: str
    company_id: str

class CustomerCreate(CamelModel):
    name: str

class CustomerResponse(CustomerBase):
    id: UUID
    created_at: datetime
    created_by: UUID
    deleted_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class CustomerWithFoldersResponse(CustomerResponse):
    folders: List[FolderResponse] = Field(default_factory=list)
