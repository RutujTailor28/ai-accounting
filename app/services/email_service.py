"""
Email service for fineyuktAI.
Handles: demo request notifications to admin + confirmation emails with .ics calendar invites to users.
Uses Gmail SMTP (free). Configure via SMTP_* env vars in .env
"""
import smtplib
import ssl
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.config import settings


# ─────────────────────────────────────────────
# ICS Calendar invite builder
# ─────────────────────────────────────────────
def build_ics(
    summary: str,
    description: str,
    organizer_email: str,
    organizer_name: str,
    attendee_email: str,
    attendee_name: str,
    start_dt: datetime,
    duration_minutes: int = 30,
    location: str = "",
) -> str:
    """Generate a .ics iCalendar event string."""
    end_dt = start_dt + timedelta(minutes=duration_minutes)

    def fmt(dt: datetime) -> str:
        # Format in UTC for maximum compatibility
        utc = dt.astimezone(timezone.utc)
        return utc.strftime("%Y%m%dT%H%M%SZ")

    uid = str(uuid.uuid4())
    now_str = fmt(datetime.now(timezone.utc))

    ics = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//fineyuktAI//Demo//EN\r\n"
        "CALSCALE:GREGORIAN\r\n"
        "METHOD:REQUEST\r\n"
        "BEGIN:VEVENT\r\n"
        f"UID:{uid}\r\n"
        f"DTSTAMP:{now_str}\r\n"
        f"DTSTART:{fmt(start_dt)}\r\n"
        f"DTEND:{fmt(end_dt)}\r\n"
        f"SUMMARY:{summary}\r\n"
        f"DESCRIPTION:{description.replace(chr(10), '\\n')}\r\n"
        f"LOCATION:{location}\r\n"
        f"ORGANIZER;CN={organizer_name}:mailto:{organizer_email}\r\n"
        f"ATTENDEE;CN={attendee_name};RSVP=TRUE:mailto:{attendee_email}\r\n"
        "STATUS:CONFIRMED\r\n"
        "SEQUENCE:0\r\n"
        "BEGIN:VALARM\r\n"
        "TRIGGER:-PT15M\r\n"
        "ACTION:DISPLAY\r\n"
        "DESCRIPTION:Reminder\r\n"
        "END:VALARM\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )
    return ics


# ─────────────────────────────────────────────
# Core SMTP sender
# ─────────────────────────────────────────────
def _send_email(msg: MIMEMultipart) -> None:
    """Send email via configured SMTP server (defaults to Gmail)."""
    host = settings.smtp_host
    port = settings.smtp_port
    user = settings.smtp_user
    password = settings.smtp_password

    if not user or not password:
        print("[WARNING] SMTP not configured. Email skipped. Set SMTP_USER and SMTP_PASSWORD in .env")
        return

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(host, port, context=context) as server:
            server.login(user, password)
            server.sendmail(msg["From"], msg["To"].split(","), msg.as_string())
        print(f"[EMAIL] Sent to {msg['To']}")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        raise


# ─────────────────────────────────────────────
# Outbound email functions
# ─────────────────────────────────────────────

def send_demo_request_to_admin(
    requester_name: str,
    requester_email: str,
    requester_company: str,
    requester_phone: Optional[str],
    requester_message: Optional[str],
    request_id: str,
) -> None:
    """
    Notify the fineyuktAI admin that a new demo has been requested.
    Includes a link to approve the demo from the admin panel.
    """
    admin_email = settings.admin_email or settings.smtp_user
    if not admin_email:
        print("[WARNING] ADMIN_EMAIL not configured. Skipping admin notification.")
        return

    backend_base = settings.backend_base_url or "http://localhost:8000"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🗓️ New Demo Request from {requester_name} ({requester_company})"
    msg["From"] = f"fineyuktAI <{settings.smtp_user}>"
    msg["To"] = admin_email

    html = f"""
    <html><body style="font-family:Inter,sans-serif;background:#f8fafc;padding:32px;">
      <div style="max-width:600px;margin:auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08);">
        <div style="background:#1d4ed8;padding:24px 32px;">
          <h1 style="color:white;margin:0;font-size:20px;">New Demo Request</h1>
        </div>
        <div style="padding:32px;">
          <table style="width:100%;border-collapse:collapse;">
            <tr><td style="padding:8px 0;color:#64748b;width:130px;">Name</td><td style="padding:8px 0;font-weight:600;">{requester_name}</td></tr>
            <tr><td style="padding:8px 0;color:#64748b;">Email</td><td style="padding:8px 0;"><a href="mailto:{requester_email}">{requester_email}</a></td></tr>
            <tr><td style="padding:8px 0;color:#64748b;">Company</td><td style="padding:8px 0;">{requester_company}</td></tr>
            <tr><td style="padding:8px 0;color:#64748b;">Phone</td><td style="padding:8px 0;">{requester_phone or '—'}</td></tr>
            <tr><td style="padding:8px 0;color:#64748b;vertical-align:top;">Message</td><td style="padding:8px 0;">{requester_message or '—'}</td></tr>
            <tr><td style="padding:8px 0;color:#64748b;">Request ID</td><td style="padding:8px 0;font-family:monospace;font-size:12px;">{request_id}</td></tr>
          </table>
          <div style="margin-top:32px;text-align:center;">
            <a href="{settings.frontend_base_url}/admin" style="display:inline-block;padding:12px 24px;background:#1d4ed8;color:white;text-decoration:none;border-radius:6px;font-weight:500;">Go to Admin Panel to Approve</a>
          </div>
        </div>
      </div>
    </body></html>
    """
    msg.attach(MIMEText(html, "html"))
    _send_email(msg)


def send_demo_confirmation_to_user(
    requester_name: str,
    requester_email: str,
    meeting_datetime: datetime,
    duration_minutes: int,
    meeting_link: Optional[str],
    admin_note: Optional[str],
) -> None:
    """
    Send the user a confirmation email with a .ics calendar invite attached.
    This goes to both the user and the admin inbox so both get the calendar event.
    """
    admin_email = settings.admin_email or settings.smtp_user
    if not settings.smtp_user:
        print("[WARNING] SMTP not configured. Skipping user confirmation.")
        return

    # Build .ics
    location = meeting_link or "Online"
    description = (
        f"Your fineyuktAI product demo has been confirmed!\n\n"
        f"Join here: {meeting_link or 'A link will be shared shortly.'}\n\n"
        + (f"Note from team: {admin_note}\n\n" if admin_note else "")
        + "See you soon!\n— The fineyuktAI Team"
    )
    ics_content = build_ics(
        summary="fineyuktAI Product Demo",
        description=description,
        organizer_email=admin_email or settings.smtp_user,
        organizer_name="fineyuktAI Team",
        attendee_email=requester_email,
        attendee_name=requester_name,
        start_dt=meeting_datetime,
        duration_minutes=duration_minutes,
        location=location,
    )

    time_str = meeting_datetime.strftime("%A, %d %B %Y at %I:%M %p")

    for recipient_email, recipient_name in [
        (requester_email, requester_name),
        (admin_email, "fineyuktAI Team"),
    ]:
        if not recipient_email:
            continue

        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"✅ Your fineyuktAI Demo is Confirmed – {time_str}"
        msg["From"] = f"fineyuktAI <{settings.smtp_user}>"
        msg["To"] = recipient_email

        html = f"""
        <html><body style="font-family:Inter,sans-serif;background:#f8fafc;padding:32px;">
          <div style="max-width:600px;margin:auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08);">
            <div style="background:#1d4ed8;padding:24px 32px;">
              <h1 style="color:white;margin:0;font-size:20px;">Your Demo is Confirmed! 🎉</h1>
            </div>
            <div style="padding:32px;">
              <p style="color:#0f172a;font-size:16px;">Hi {recipient_name if recipient_email == requester_email else 'Team'},</p>
              <p style="color:#475569;">Your fineyuktAI product demo has been scheduled. Here are the details:</p>
              <div style="background:#f1f5f9;border-radius:10px;padding:20px;margin:24px 0;">
                <p style="margin:0 0 8px;"><strong>📅 Date &amp; Time:</strong> {time_str}</p>
                <p style="margin:0 0 8px;"><strong>⏱️ Duration:</strong> {duration_minutes} minutes</p>
                <p style="margin:0;"><strong>🔗 Meeting Link:</strong> {'<a href="' + meeting_link + '" style="color:#1d4ed8;">' + meeting_link + '</a>' if meeting_link else 'Will be shared shortly'}</p>
                {'<p style="margin:8px 0 0;"><strong>📝 Note:</strong> ' + admin_note + '</p>' if admin_note else ''}
              </div>
              <p style="color:#475569;font-size:14px;">A calendar invite (.ics file) is attached — click it to add this event directly to your calendar (Google, Outlook, Apple).</p>
              <div style="margin-top:32px;padding-top:24px;border-top:1px solid #e2e8f0;color:#94a3b8;font-size:12px;">
                <p>fineyuktAI · AI-Driven Accounting Platform<br>
                Reply to this email if you need to reschedule.</p>
              </div>
            </div>
          </div>
        </body></html>
        """

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(html, "html"))
        msg.attach(alt)

        # Attach .ics file
        ics_part = MIMEBase("text", "calendar", method="REQUEST")
        ics_part.set_payload(ics_content.encode("utf-8"))
        encoders.encode_base64(ics_part)
        ics_part.add_header("Content-Disposition", "attachment", filename="fineyuktai_demo.ics")
        ics_part.add_header("Content-Type", "text/calendar; method=REQUEST; charset=UTF-8")
        msg.attach(ics_part)

        _send_email(msg)
