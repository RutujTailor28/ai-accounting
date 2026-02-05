from typing import Optional, List
from datetime import datetime
from .base import CamelModel

class RoleCreate(CamelModel):
    name: str
    description: Optional[str] = None
    company_id: Optional[str] = None
    permissions: List[str] = []

class RoleUpdate(CamelModel):
    name: Optional[str] = None
    description: Optional[str] = None
    company_id: Optional[str] = None
    permissions: Optional[List[str]] = None

class RoleResponse(CamelModel):
    id: int
    name: str
    description: Optional[str] = None
    company_id: Optional[str] = None
    permissions: List[str] = []
    created_at: datetime
