from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class DemoRequestCreate(BaseModel):
    """Submitted by a visitor on the landing page."""
    name: str
    email: EmailStr
    company: str
    phone: Optional[str] = None
    message: Optional[str] = None

class DemoRequestResponse(BaseModel):
    """Stored demo request record."""
    id: str
    name: str
    email: str
    company: str
    phone: Optional[str] = None
    message: Optional[str] = None
    status: str  # "pending" | "approved" | "rejected"
    created_at: str
    meeting_datetime: Optional[str] = None
    duration_minutes: Optional[int] = None
    meeting_link: Optional[str] = None
    admin_note: Optional[str] = None

class DemoApprove(BaseModel):
    """Admin provides meeting datetime when approving."""
    meeting_datetime: datetime   # ISO 8601, e.g. "2026-09-01T11:00:00+05:30"
    duration_minutes: int = 30
    meeting_link: Optional[str] = None  # e.g. Google Meet / Zoom link
    admin_note: Optional[str] = None
