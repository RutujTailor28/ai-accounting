from typing import Optional
import datetime
from app.schemas.base import CamelModel

class CompanyBase(CamelModel):
    name: str

class CompanyUpdate(CamelModel):
    name: Optional[str] = None

class CompanyResponse(CompanyBase):
    id: str
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

    class Config:
        from_attributes = True
