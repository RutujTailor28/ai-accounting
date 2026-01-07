from typing import List, Optional
from datetime import datetime
from pydantic import Field
from .base import CamelModel


class UserCreate(CamelModel):
    """Schema for creating a new user."""

    email: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company_id: Optional[str] = None
    roles: List[str] = Field(default_factory=lambda: ["user"])


class UserUpdate(CamelModel):
    """Schema for updating an existing user."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    roles: Optional[List[str]] = None


class UserResponse(CamelModel):
    """Schema returned from user APIs."""

    id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company_id: Optional[str] = None
    created_at: datetime
    roles: List[str] = Field(default_factory=list)
