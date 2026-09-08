"""Public trial route - no login required.

Runs the Phase 1 deterministic pipeline only. There is no LLM call anywhere
in this module, which is what makes it safe to expose: a public endpoint that
invoked a model would be an uncapped bill waiting to happen.

Because the pipeline is pure Python it completes in milliseconds even for a
few thousand rows, so every endpoint here is synchronous. No queue, no
worker, no serverless timeout problem.

Guarded by a shared access code rather than accounts - enough to keep the
open internet out, while an accountant only needs a link.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.phase1 import coa, trial_store
from app.phase1.contracts import money
from app.phase1.pipeline import rebuild, run

router = APIRouter()

ALLOWED_EXTENSIONS = (".pdf", ".csv", ".xlsx", ".xls")


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class SessionRequest(BaseModel):
    code: str = Field(..., max_length=64)


class SessionResponse(BaseModel):
    session_id: str
    max_uploads: int
    max_file_mb: int


class ReclassifyRequest(BaseModel):
    session_id: str
    document_id: str
    # row_index -> ledger code
    overrides: Dict[int, str] = Field(default_factory=dict)
    opening_balance: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str
    document_id: Optional[str] = None
    verdict: Optional[str] = Field(None, max_length=64)
    what_is_wrong: Optional[str] = Field(None, max_length=5000)
    misclassified_rows: Optional[str] = Field(None, max_length=5000)
    would_use: Optional[str] = Field(None, max_length=64)
    contact: Optional[str] = Field(None, max_length=200)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _require_session(session_id: str) -> Dict[str, Any]:
    session = trial_store.get_session(session_id)
    if not trial_store.session_is_live(session):
        raise HTTPException(status_code=404, detail="Session not found or expired. Reload the page to start again.")
    return session  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@router.post("/session", response_model=SessionResponse)
async def start_session(body: SessionRequest, request: Request) -> SessionResponse:
    """Exchange an access code for a session id."""
    if not trial_store.code_is_valid(body.code):
        raise HTTPException(status_code=403, detail="Invalid access code.")

    ip = _client_ip(request)
    if not trial_store.ip_within_limit(ip):
        raise HTTPException(
            status_code=429,
            detail="Daily limit reached for this network. Please try again tomorrow.",
        )

    session_id = trial_store.create_session(
        code=body.code,
        ip=ip,
        user_agent=request.headers.get("user-agent", ""),
    )
    return SessionResponse(
        session_id=session_id,
        max_uploads=trial_store.MAX_UPLOADS_PER_SESSION,
        max_file_mb=trial_store.MAX_UPLOAD_BYTES // (1024 * 1024),
    )


@router.get("/chart")
async def chart_of_accounts() -> Dict[str, Any]:
    """Ledger heads for the reclassify dropdown."""
    return {"accounts": coa.selectable_heads(), "suspense_code": coa.SUSPENSE}


@router.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    opening_balance: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Upload a bank statement and get statements back immediately."""
    _require_session(session_id)

    if trial_store.upload_count(session_id) >= trial_store.MAX_UPLOADS_PER_SESSION:
        raise HTTPException(
            status_code=429,
            detail=f"This session is limited to {trial_store.MAX_UPLOADS_PER_SESSION} files.",
        )

    filename = file.filename or "upload"
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF, XLSX, XLS or CSV bank statement.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="That file is empty.")
    if len(content) > trial_store.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {trial_store.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    opening = money(opening_balance) if opening_balance else None

    try:
        result = run(content, filename, opening_balance=opening, scope=session_id)
    except Exception as exc:  # never leak a stack trace to a public caller
        print(f"[TRIAL][ERROR] pipeline crashed on {filename}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Could not process that statement. Please try a different export format.",
        )

    if not result.ok:
        # A rejected file is an expected outcome, not a server error: the
        # balance gate did its job. Return 200 with the reason so the UI can
        # explain it.
        return {
            "ok": False,
            "document_id": None,
            "errors": result.errors,
            "warnings": result.warnings,
            "rows_found": len(result.transactions),
        }

    payload = result.as_dict()
    doc_id = trial_store.save_document(
        session_id=session_id,
        filename=filename,
        transactions=[t.to_dict() for t in result.transactions],
        result=payload,
        opening_balance=str(result.statements.opening_balance if result.statements else "0.00"),
    )

    payload["document_id"] = doc_id
    payload["filename"] = filename
    return payload


@router.post("/reclassify")
async def reclassify(body: ReclassifyRequest) -> Dict[str, Any]:
    """Re-run the arithmetic after a reviewer changes some ledgers.

    Re-uses the stored transactions rather than re-parsing, so corrections
    cannot be lost to a different parse of the same file.
    """
    _require_session(body.session_id)

    doc = trial_store.get_document(body.session_id, body.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    unknown = [c for c in body.overrides.values() if c not in coa.CHART]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown ledger code(s): {unknown}")

    opening = money(body.opening_balance) if body.opening_balance else money(
        doc.get("opening_balance") or "0.00"
    )

    try:
        result = rebuild(doc["transactions"], opening_balance=opening, overrides=body.overrides)
    except Exception as exc:
        print(f"[TRIAL][ERROR] reclassify failed: {exc}")
        raise HTTPException(status_code=500, detail="Could not rebuild the statements.")

    payload = result.as_dict()
    payload["document_id"] = body.document_id
    trial_store.update_document_result(
        body.session_id,
        body.document_id,
        [t.to_dict() for t in result.transactions],
        payload,
    )
    return payload


@router.post("/feedback")
async def feedback(body: FeedbackRequest) -> Dict[str, str]:
    """Capture the accountant's verdict. The whole point of the pilot."""
    _require_session(body.session_id)
    trial_store.save_feedback(
        body.session_id,
        {
            "document_id": body.document_id,
            "verdict": body.verdict,
            "what_is_wrong": body.what_is_wrong,
            "misclassified_rows": body.misclassified_rows,
            "would_use": body.would_use,
            "contact": body.contact,
        },
    )
    return {"status": "recorded", "message": "Thank you - this is exactly what we need."}
