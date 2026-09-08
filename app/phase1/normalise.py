"""Narration normaliser - pure Python, no LLM, no network.

Turns a raw bank narration into `(clean_text, counterparty_key, channel)`.

    UPI/423122011/Payment from/SWIGGY@axisbank/UTIB0000123
        -> ("payment from swiggy", "swiggy", Channel.UPI)

This function decides how many rows the rules engine can match, which decides
how many rows would ever need a model. It is worth more test cases than any
other single function in the pipeline.
"""

from __future__ import annotations

import re
from typing import Tuple

from app.phase1.contracts import Channel

# Channel detection. Order matters - first hit wins, so put the specific
# markers ahead of the generic ones.
_CHANNEL_PATTERNS: list[tuple[Channel, re.Pattern]] = [
    (Channel.UPI, re.compile(r"\b(upi|vpa|bhim|gpay|phonepe|paytm[\s-]?upi)\b", re.I)),
    (Channel.NEFT, re.compile(r"\bneft\b", re.I)),
    (Channel.RTGS, re.compile(r"\brtgs\b", re.I)),
    (Channel.IMPS, re.compile(r"\bimps\b", re.I)),
    (Channel.CHEQUE, re.compile(r"\b(chq|cheque|clg|clearing|micr)\b", re.I)),
    (Channel.ATM, re.compile(r"\b(atm|cash\s*wdl|cash\s*withdrawal|nwd)\b", re.I)),
    (Channel.POS, re.compile(r"\b(pos|ecom|e-?com|card\s*payment|debit\s*card)\b", re.I)),
    (Channel.INTEREST, re.compile(r"\b(int\.?\s*(pd|cr|coll)|interest|intt)\b", re.I)),
    (Channel.CHARGES, re.compile(
        r"\b(chrg|charges?|chgs|fee|comm|commission|penalty|amc|sms\s*chg|gst\s*on)\b", re.I)),
    (Channel.CASH, re.compile(r"\b(cash\s*dep|cdm|by\s*cash|to\s*cash)\b", re.I)),
    (Channel.TRANSFER, re.compile(r"\b(trf|transfer|ft|funds?\s*tran)\b", re.I)),
]

# Noise stripped before extracting the counterparty.
_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b\d{6,}\b"),                      # UTR / ref / txn ids
    re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),        # IFSC codes
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),  # embedded dates
    re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b"),      # times
    re.compile(r"\bX{2,}\d+\b", re.I),              # masked account numbers
    re.compile(r"\b\d{2,}[A-Z]{2,}\d{2,}\b"),       # mixed reference blobs
]

# Words that are never a counterparty name.
_STOPWORDS = {
    "upi", "neft", "imps", "rtgs", "chq", "cheque", "clg", "atm", "pos", "trf",
    "transfer", "payment", "paid", "pay", "from", "to", "by", "ref", "no",
    "txn", "tran", "transaction", "acc", "ac", "account", "bank", "ltd", "pvt",
    "the", "and", "for", "of", "in", "on", "at", "via", "dr", "cr", "inb",
    "ib", "mob", "net", "banking", "online", "self", "sent", "received", "rcvd",
    "credit", "debit", "deposit", "withdrawal", "charges", "charge", "fee",
    "india", "collect", "request", "success", "successful", "money",
}

# Bank-specific prefixes worth removing wholesale.
_PREFIX_RE = re.compile(
    r"^(upi|neft|imps|rtgs|chq|clg|pos|atm|ecom|mmt|ift|inb|ib|mob|ach|nach)[\s/:-]+",
    re.I,
)

_SPLIT_RE = re.compile(r"[/|\\:;,]+")
_NONWORD_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")


def detect_channel(narration: str) -> Channel:
    """Classify how the money moved. Cheap and independent of amount."""
    if not narration:
        return Channel.OTHER
    for channel, pattern in _CHANNEL_PATTERNS:
        if pattern.search(narration):
            return channel
    return Channel.OTHER


def _strip_noise(text: str) -> str:
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def _extract_vpa_name(narration: str) -> str:
    """For UPI, the VPA usually carries the cleanest counterparty name.

    `swiggy@axisbank` -> `swiggy`. A purely numeric handle (a phone number
    used as a VPA) carries no name, so it is rejected.
    """
    match = re.search(r"([A-Za-z0-9._-]{2,})@[A-Za-z]{2,}", narration)
    if not match:
        return ""
    handle = match.group(1)
    if re.fullmatch(r"[\d.\-_]+", handle):
        return ""
    handle = re.sub(r"[._-]+", " ", handle)
    handle = re.sub(r"\d{4,}", " ", handle)
    return handle.strip()


def _best_segment(narration: str) -> str:
    """Pick the segment most likely to hold a counterparty name.

    Bank narrations are delimiter-salad; the longest segment with the most
    alphabetic content is a reliable heuristic.
    """
    segments = [s.strip() for s in _SPLIT_RE.split(narration) if s.strip()]
    best, best_score = "", 0.0
    for seg in segments:
        letters = sum(c.isalpha() for c in seg)
        if letters < 3:
            continue
        words = [w for w in _NONWORD_RE.sub(" ", seg.lower()).split() if w not in _STOPWORDS]
        if not words:
            continue
        score = letters * (1.0 + 0.1 * len(words))
        if score > best_score:
            best, best_score = seg, score
    return best


def _to_key(text: str) -> str:
    """lowercase, alphanumerics + single spaces, stopwords dropped."""
    text = _NONWORD_RE.sub(" ", text.lower())
    words = [w for w in text.split() if w and w not in _STOPWORDS and not w.isdigit()]
    # Drop 1-character fragments left behind by stripping.
    words = [w for w in words if len(w) > 1]
    return _WS_RE.sub(" ", " ".join(words)).strip()


def normalise(narration: str) -> Tuple[str, str, Channel]:
    """Return `(narration_clean, counterparty_key, channel)`.

    `counterparty_key` may be empty - that is a legitimate outcome for rows
    like "SMS CHARGES", which have a channel but no counterparty.
    """
    if not narration or not narration.strip():
        return "", "", Channel.OTHER

    raw = narration.strip()
    channel = detect_channel(raw)

    working = _strip_noise(raw)
    working = _PREFIX_RE.sub("", working)

    counterparty = ""
    if channel == Channel.UPI:
        counterparty = _extract_vpa_name(raw)
    if not counterparty:
        counterparty = _best_segment(working)

    key = _to_key(counterparty)
    # Keep keys short - long keys are usually leftover reference blobs and
    # match nothing.
    if key:
        key = " ".join(key.split()[:4])

    clean = _to_key(working)
    clean = " ".join(clean.split()[:12])

    return clean, key, channel
