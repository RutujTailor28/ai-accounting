"""
Demo Booking API endpoints.

POST /api/v1/demo/request           → User submits demo request (no auth needed)
POST /api/v1/demo/{id}/approve      → Admin approves & schedules meeting (requires admin auth)
GET  /api/v1/demo/requests          → Admin lists all demo requests (requires admin auth)
"""
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from app.schemas.demo import DemoRequestCreate, DemoRequestResponse, DemoApprove
from app.core.supabase import supabase_admin
from app.services.email_service import send_demo_request_to_admin, send_demo_confirmation_to_user
from app.api.deps import get_current_user

router = APIRouter()

TABLE = "demo_requests"


# ─────────────────────────────────────────────
# Public: Submit a demo request
# ─────────────────────────────────────────────
@router.post("/request", response_model=DemoRequestResponse, status_code=201)
async def submit_demo_request(
    data: DemoRequestCreate,
    background_tasks: BackgroundTasks,
):
    """
    Public endpoint — no auth required.
    Stores the demo request in Supabase and emails fineyukt admin.
    """
    request_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    record = {
        "id": request_id,
        "name": data.name,
        "email": data.email.lower().strip(),
        "company": data.company,
        "phone": data.phone or None,
        "message": data.message or None,
        "status": "pending",
        "created_at": created_at,
    }

    try:
        supabase_admin.table(TABLE).insert(record).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save demo request: {str(e)}")

    # Email admin in background (non-blocking)
    background_tasks.add_task(
        send_demo_request_to_admin,
        requester_name=data.name,
        requester_email=data.email,
        requester_company=data.company,
        requester_phone=data.phone,
        requester_message=data.message,
        request_id=request_id,
    )

    return DemoRequestResponse(**record)


# ─────────────────────────────────────────────
# Admin: Approve a demo request
# ─────────────────────────────────────────────
@router.post("/{request_id}/approve", response_model=dict)
async def approve_demo_request(
    request_id: str,
    data: DemoApprove,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    """
    Admin-only endpoint.
    Marks the request as approved, then sends the user a confirmation
    email with a .ics calendar invite attached.
    """
    # Fetch the request
    try:
        result = supabase_admin.table(TABLE).select("*").eq("id", request_id).single().execute()
    except Exception:
        raise HTTPException(status_code=404, detail="Demo request not found")

    if not result.data:
        raise HTTPException(status_code=404, detail="Demo request not found")

    demo = result.data

    if demo["status"] == "approved":
        raise HTTPException(status_code=400, detail="This demo request is already approved")

    # Update status
    supabase_admin.table(TABLE).update({
        "status": "approved",
        "meeting_datetime": data.meeting_datetime.isoformat(),
        "meeting_link": data.meeting_link,
        "duration_minutes": data.duration_minutes,
        "admin_note": data.admin_note,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": getattr(current_user, "id", current_user.get("id") if isinstance(current_user, dict) else None),
    }).eq("id", request_id).execute()

    # Send confirmation email + .ics to user (and admin cc) in background
    background_tasks.add_task(
        send_demo_confirmation_to_user,
        requester_name=demo["name"],
        requester_email=demo["email"],
        meeting_datetime=data.meeting_datetime,
        duration_minutes=data.duration_minutes,
        meeting_link=data.meeting_link,
        admin_note=data.admin_note,
    )

    return {
        "message": "Demo approved. Confirmation email with calendar invite sent to user.",
        "request_id": request_id,
        "meeting_datetime": data.meeting_datetime.isoformat(),
    }


# ─────────────────────────────────────────────
# Admin: List all demo requests
# ─────────────────────────────────────────────
@router.get("/requests", response_model=List[DemoRequestResponse])
async def list_demo_requests(
    current_user: dict = Depends(get_current_user),
):
    """Admin-only: List all demo requests sorted by newest first."""
    try:
        result = supabase_admin.table(TABLE).select("*").order("created_at", desc=True).execute()
        return [DemoRequestResponse(**r) for r in (result.data or [])]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
