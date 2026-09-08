"""Storage for the public trial route.

Persists sessions, parsed transactions and feedback. Supabase is the primary
store; an in-process fallback keeps local development working before the
migration is applied. The fallback is per-process, so it is fine for a dev
machine and useless on serverless - which is exactly why the Supabase tables
must exist before the link goes to anyone.

Only *parsed transactions* are persisted, never the uploaded file. That keeps
reclassification cheap and means the raw statement need not live in the
database.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Access codes that unlock the trial. Comma-separated in the environment.
# Not authentication - just a doormat that stops the open internet walking in.
TRIAL_CODES = {
    c.strip().upper()
    for c in os.getenv("TRIAL_ACCESS_CODES", "PILOT-2026").split(",")
    if c.strip()
}

MAX_UPLOAD_BYTES = int(os.getenv("TRIAL_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_UPLOADS_PER_SESSION = int(os.getenv("TRIAL_MAX_UPLOADS", "5"))
MAX_SESSIONS_PER_IP_PER_DAY = int(os.getenv("TRIAL_MAX_SESSIONS_PER_IP", "10"))
SESSION_TTL_HOURS = int(os.getenv("TRIAL_SESSION_TTL_HOURS", "72"))

_lock = threading.Lock()
_memory: Dict[str, Dict[str, Any]] = {}
_ip_counter: Dict[str, List[float]] = {}


def _supabase():
    """Return the admin client, or None when Supabase is not configured."""
    try:
        from app.core.supabase import supabase_admin

        return supabase_admin
    except Exception:
        return None


def code_is_valid(code: str) -> bool:
    return bool(code) and code.strip().upper() in TRIAL_CODES


def ip_within_limit(ip: str) -> bool:
    """Crude per-IP throttle. Good enough for a pilot with a handful of users."""
    if not ip:
        return True
    now = time.time()
    cutoff = now - 86400
    with _lock:
        hits = [t for t in _ip_counter.get(ip, []) if t > cutoff]
        if len(hits) >= MAX_SESSIONS_PER_IP_PER_DAY:
            _ip_counter[ip] = hits
            return False
        hits.append(now)
        _ip_counter[ip] = hits
    return True


def create_session(code: str, ip: str = "", user_agent: str = "") -> str:
    session_id = "trial_" + secrets.token_urlsafe(16)
    record = {
        "id": session_id,
        "access_code": code.strip().upper(),
        "ip": ip[:64],
        "user_agent": user_agent[:300],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (
            datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)
        ).isoformat(),
        "upload_count": 0,
    }

    client = _supabase()
    if client is not None:
        try:
            client.table("trial_sessions").insert(record).execute()
            return session_id
        except Exception as exc:
            print(f"[TRIAL][WARN] Supabase insert failed, using memory: {exc}")

    with _lock:
        _memory[session_id] = {"session": record, "documents": {}, "feedback": []}
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    if not session_id:
        return None

    client = _supabase()
    if client is not None:
        try:
            res = (
                client.table("trial_sessions")
                .select("*")
                .eq("id", session_id)
                .limit(1)
                .execute()
            )
            if res.data:
                return res.data[0]
        except Exception as exc:
            print(f"[TRIAL][WARN] Supabase read failed: {exc}")

    with _lock:
        entry = _memory.get(session_id)
    return entry["session"] if entry else None


def session_is_live(session: Optional[Dict[str, Any]]) -> bool:
    if not session:
        return False
    expires = session.get("expires_at")
    if not expires:
        return True
    try:
        return datetime.fromisoformat(str(expires)) > datetime.now(timezone.utc)
    except Exception:
        return True


def save_document(
    session_id: str,
    filename: str,
    transactions: List[Dict[str, Any]],
    result: Dict[str, Any],
    opening_balance: str,
) -> str:
    doc_id = "doc_" + secrets.token_urlsafe(10)
    record = {
        "id": doc_id,
        "session_id": session_id,
        "filename": filename[:255],
        "transactions": transactions,
        "result": result,
        "opening_balance": opening_balance,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    client = _supabase()
    if client is not None:
        try:
            client.table("trial_documents").insert(record).execute()
            client.table("trial_sessions").update(
                {"upload_count": (get_session(session_id) or {}).get("upload_count", 0) + 1}
            ).eq("id", session_id).execute()
            return doc_id
        except Exception as exc:
            print(f"[TRIAL][WARN] Supabase doc insert failed, using memory: {exc}")

    with _lock:
        entry = _memory.setdefault(
            session_id, {"session": {"id": session_id}, "documents": {}, "feedback": []}
        )
        entry["documents"][doc_id] = record
        entry["session"]["upload_count"] = len(entry["documents"])
    return doc_id


def get_document(session_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
    client = _supabase()
    if client is not None:
        try:
            res = (
                client.table("trial_documents")
                .select("*")
                .eq("id", doc_id)
                .eq("session_id", session_id)
                .limit(1)
                .execute()
            )
            if res.data:
                row = res.data[0]
                # JSONB may come back as a string depending on client version.
                for key in ("transactions", "result"):
                    if isinstance(row.get(key), str):
                        row[key] = json.loads(row[key])
                return row
        except Exception as exc:
            print(f"[TRIAL][WARN] Supabase doc read failed: {exc}")

    with _lock:
        entry = _memory.get(session_id)
        return entry["documents"].get(doc_id) if entry else None


def update_document_result(
    session_id: str, doc_id: str, transactions: List[Dict[str, Any]], result: Dict[str, Any]
) -> None:
    client = _supabase()
    if client is not None:
        try:
            client.table("trial_documents").update(
                {"transactions": transactions, "result": result}
            ).eq("id", doc_id).eq("session_id", session_id).execute()
            return
        except Exception as exc:
            print(f"[TRIAL][WARN] Supabase doc update failed: {exc}")

    with _lock:
        entry = _memory.get(session_id)
        if entry and doc_id in entry["documents"]:
            entry["documents"][doc_id]["transactions"] = transactions
            entry["documents"][doc_id]["result"] = result


def upload_count(session_id: str) -> int:
    session = get_session(session_id)
    return int((session or {}).get("upload_count", 0) or 0)


def save_feedback(session_id: str, payload: Dict[str, Any]) -> None:
    record = {
        "session_id": session_id,
        "payload": payload,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    client = _supabase()
    if client is not None:
        try:
            client.table("trial_feedback").insert(record).execute()
            return
        except Exception as exc:
            print(f"[TRIAL][WARN] Supabase feedback insert failed: {exc}")

    with _lock:
        entry = _memory.setdefault(
            session_id, {"session": {"id": session_id}, "documents": {}, "feedback": []}
        )
        entry["feedback"].append(record)
