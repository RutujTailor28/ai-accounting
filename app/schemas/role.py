from typing import Optional
from datetime import datetime
from .base import CamelModel

class RoleCreate(CamelModel):
    name: str
    description: Optional[str] = None
    company_id: Optional[str] = None

class RoleUpdate(CamelModel):
    name: Optional[str] = None
    description: Optional[str] = None
    company_id: Optional[str] = None

class RoleResponse(CamelModel):
    id: int
    name: str
    description: Optional[str] = None
    company_id: Optional[str] = None
    created_at: datetime
