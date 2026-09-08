"""Bank statement parsers - structured rows, not a wall of text.

The existing `DocumentParser` returns flat text, which is why an LLM is
currently needed to recover transactions. These parsers pull *tables* and map
their columns onto the canonical `Transaction`, so no model is involved.

Strategy is format-driven rather than bank-driven: header detection covers
most Indian bank exports without a per-bank adapter. A bank whose layout
defeats it gets an explicit adapter later - but only after a real statement
proves it necessary.
"""

from __future__ import annotations

import io
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, List, Optional, Sequence

from app.phase1.contracts import (
    ZERO,
    ParseResult,
    Transaction,
    money,
)
from app.phase1.normalise import normalise

# Column header synonyms seen across Indian bank exports.
_DATE_HEADERS = ("date", "txn date", "transaction date", "value date", "tran date", "post date")
_NARRATION_HEADERS = (
    "narration", "particulars", "description", "transaction remarks", "remarks",
    "details", "transaction details", "transaction", "narrative",
)
_DEBIT_HEADERS = ("withdrawal", "withdrawal amt", "debit", "debit amount", "dr", "withdrawals", "paid out")
_CREDIT_HEADERS = ("deposit", "deposit amt", "credit", "credit amount", "cr", "deposits", "paid in")
_BALANCE_HEADERS = ("balance", "closing balance", "running balance", "balance amt")
_REF_HEADERS = ("ref no", "reference", "chq no", "cheque no", "chq./ref.no.", "ref", "utr")

_DATE_FORMATS = (
    "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
    "%Y-%m-%d", "%Y/%m/%d",
    "%d %b %Y", "%d-%b-%Y", "%d %B %Y", "%d-%b-%y",
    "%m/%d/%Y", "%d.%m.%Y",
)

# How far the running balance may drift before a row is called a failure.
BALANCE_TOLERANCE = Decimal("0.05")
# Above this share of failing rows the statement is rejected outright.
BALANCE_FAIL_THRESHOLD = 0.02


def _norm_header(value: Any) -> str:
    return re.sub(r"[\s_]+", " ", str(value or "").strip().lower()).strip(" .:")


def _match_column(header: str, candidates: Sequence[str]) -> bool:
    h = _norm_header(header)
    if not h:
        return False
    return any(h == c or h.startswith(c) or c in h for c in candidates)


def parse_date(value: Any) -> Optional[date]:
    """Parse the date formats Indian banks actually emit."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    # Strip a trailing time component if present.
    text = re.sub(r"\s+\d{1,2}:\d{2}(:\d{2})?$", "", text)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


class ColumnMap:
    """Resolved column indices for one table."""

    __slots__ = ("date", "narration", "debit", "credit", "balance", "ref")

    def __init__(self) -> None:
        self.date: Optional[int] = None
        self.narration: Optional[int] = None
        self.debit: Optional[int] = None
        self.credit: Optional[int] = None
        self.balance: Optional[int] = None
        self.ref: Optional[int] = None

    @property
    def usable(self) -> bool:
        """Minimum viable statement: a date, a description and an amount."""
        return (
            self.date is not None
            and self.narration is not None
            and (self.debit is not None or self.credit is not None)
        )

    @classmethod
    def detect(cls, header_row: Sequence[Any]) -> "ColumnMap":
        cm = cls()
        for idx, cell in enumerate(header_row):
            if cm.date is None and _match_column(cell, _DATE_HEADERS):
                cm.date = idx
            elif cm.narration is None and _match_column(cell, _NARRATION_HEADERS):
                cm.narration = idx
            elif cm.debit is None and _match_column(cell, _DEBIT_HEADERS):
                cm.debit = idx
            elif cm.credit is None and _match_column(cell, _CREDIT_HEADERS):
                cm.credit = idx
            elif cm.balance is None and _match_column(cell, _BALANCE_HEADERS):
                cm.balance = idx
            elif cm.ref is None and _match_column(cell, _REF_HEADERS):
                cm.ref = idx
        return cm


def _find_header_row(rows: Sequence[Sequence[Any]], scan_limit: int = 30) -> tuple[int, ColumnMap]:
    """Locate the header row.

    Bank exports bury the table under branch details, address blocks and
    disclaimers, so the header is rarely row 0.
    """
    best_idx, best_map, best_score = -1, ColumnMap(), 0
    for idx, row in enumerate(rows[:scan_limit]):
        cm = ColumnMap.detect(row)
        score = sum(
            1 for v in (cm.date, cm.narration, cm.debit, cm.credit, cm.balance)
            if v is not None
        )
        if cm.usable and score > best_score:
            best_idx, best_map, best_score = idx, cm, score
    return best_idx, best_map


def _row_to_transaction(
    row: Sequence[Any],
    cm: ColumnMap,
    row_index: int,
    source_document: str,
) -> Optional[Transaction]:
    """Map one raw row. Returns None for rows that are not transactions."""

    def cell(idx: Optional[int]) -> Any:
        if idx is None or idx >= len(row):
            return ""
        return row[idx]

    txn_date = parse_date(cell(cm.date))
    narration = str(cell(cm.narration) or "").strip()
    narration = re.sub(r"\s+", " ", narration)

    debit = money(cell(cm.debit)) if cm.debit is not None else ZERO
    credit = money(cell(cm.credit)) if cm.credit is not None else ZERO

    # Totals/footers and blank spacer rows.
    if not txn_date and not narration:
        return None
    if narration and re.match(r"^(total|closing|opening|grand total|b/f|c/f)\b", narration, re.I):
        return None
    if debit == ZERO and credit == ZERO:
        return None

    # Some banks emit a single signed amount column instead of two.
    if debit < ZERO:
        credit, debit = -debit, ZERO
    if credit < ZERO:
        debit, credit = -credit, ZERO

    balance = money(cell(cm.balance)) if cm.balance is not None else None
    if cm.balance is not None and str(cell(cm.balance)).strip() == "":
        balance = None

    txn = Transaction(
        txn_date=txn_date,
        narration_raw=narration,
        debit=debit,
        credit=credit,
        balance=balance,
        row_index=row_index,
        source_document=source_document,
        ref_no=str(cell(cm.ref) or "").strip(),
    )
    txn.narration_clean, txn.counterparty_key, txn.channel = normalise(narration)
    return txn


def _rows_to_result(
    rows: Sequence[Sequence[Any]],
    source_document: str,
    bank_hint: str = "",
) -> ParseResult:
    result = ParseResult(source_document=source_document, bank_hint=bank_hint)

    header_idx, cm = _find_header_row(rows)
    if header_idx < 0:
        result.add_error(
            "no_header",
            "Could not find a transaction table. Expected columns like "
            "Date / Narration / Withdrawal / Deposit / Balance.",
        )
        return result

    for offset, row in enumerate(rows[header_idx + 1:], start=1):
        try:
            txn = _row_to_transaction(row, cm, offset, source_document)
        except Exception as exc:
            result.add_warning("row_error", f"Row {offset}: {exc}", offset)
            continue
        if txn is not None:
            result.transactions.append(txn)

    if not result.transactions:
        result.add_error("no_rows", "Found a table header but no transaction rows under it.")

    undated = sum(1 for t in result.transactions if t.txn_date is None)
    if undated:
        result.add_warning(
            "undated_rows", f"{undated} row(s) had an unreadable date."
        )
    return result


# --------------------------------------------------------------------------
# Format entry points
# --------------------------------------------------------------------------


def parse_tabular(content: bytes, filename: str) -> ParseResult:
    """CSV / XLS / XLSX via pandas. The most reliable path - use it when the
    accountant can export from net banking."""
    name = filename.lower()
    row_sets: List[List[List[Any]]] = []

    if name.endswith(".csv"):
        # Deliberately NOT pandas. Bank CSVs are ragged - a one-cell bank-name
        # row above a six-column table - and pandas infers its column count
        # from the first rows, then silently drops every wider row as "bad".
        # csv.reader keeps ragged rows intact.
        import csv as _csv

        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="replace")

        sample = text[:8192]
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
        except Exception:
            # Fall back to whichever candidate appears most in the sample.
            delimiter = max(",;\t|", key=sample.count)

        try:
            rows = [list(r) for r in _csv.reader(io.StringIO(text), delimiter=delimiter)]
            row_sets.append(rows)
        except Exception as exc:
            result = ParseResult(source_document=filename)
            result.add_error("read_failed", f"Could not read the CSV: {exc}")
            return result
    else:
        import pandas as pd

        try:
            sheets = pd.read_excel(io.BytesIO(content), header=None, dtype=str, sheet_name=None)
        except Exception as exc:
            result = ParseResult(source_document=filename)
            result.add_error("read_failed", f"Could not read the workbook: {exc}")
            return result
        for frame in sheets.values():
            row_sets.append(frame.where(frame.notna(), "").values.tolist())

    best: Optional[ParseResult] = None
    for rows in row_sets:
        candidate = _rows_to_result(rows, filename)
        if best is None or len(candidate.transactions) > len(best.transactions):
            best = candidate

    return best or ParseResult(source_document=filename)


def parse_pdf(content: bytes, filename: str) -> ParseResult:
    """PDF via pdfplumber table extraction.

    Note this is `extract_table`, not `extract_text` - the whole point of
    Phase 1 is to recover structure rather than hand prose to a model.
    """
    import pdfplumber

    result = ParseResult(source_document=filename)
    all_rows: List[List[Any]] = []

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables or []:
                    for row in table:
                        all_rows.append([("" if c is None else str(c).replace("\n", " ")) for c in row])
    except Exception as exc:
        result.add_error("pdf_failed", f"Could not read the PDF: {exc}")
        return result

    if not all_rows:
        result.add_error(
            "no_tables",
            "No tables found in this PDF. It is probably a scanned image - "
            "please upload the Excel or CSV export from net banking instead.",
        )
        return result

    parsed = _rows_to_result(all_rows, filename)
    # A repeated header on every page means the map is re-detected per page;
    # keeping all rows and detecting once is correct because rows above the
    # first header are skipped and later headers fail the row filters.
    return parsed


def parse(content: bytes, filename: str) -> ParseResult:
    """Dispatch on extension."""
    name = (filename or "").lower()
    if name.endswith((".csv", ".xlsx", ".xls")):
        return parse_tabular(content, filename)
    if name.endswith(".pdf"):
        return parse_pdf(content, filename)

    result = ParseResult(source_document=filename)
    result.add_error(
        "unsupported",
        f"Unsupported file type. Upload a PDF, XLSX, XLS or CSV bank statement.",
    )
    return result


# --------------------------------------------------------------------------
# The validation gate
# --------------------------------------------------------------------------


def validate_running_balance(result: ParseResult) -> ParseResult:
    """Check every row against the statement's own running balance.

    `balance[i] == balance[i-1] - debit + credit`

    This is the single most valuable check in the pipeline: it catches
    mis-parsed columns, dropped rows and OCR digit errors *before* they reach
    the ledger. A statement that fails it is rejected rather than silently
    producing wrong accounts.
    """
    rows = [t for t in result.transactions if t.balance is not None]
    if len(rows) < 2:
        result.balance_checked = False
        result.add_warning(
            "no_balance_column",
            "This statement has no running balance column, so rows could not "
            "be cross-checked. Figures may be incomplete.",
        )
        return result

    failures = 0
    for prev, curr in zip(rows, rows[1:]):
        expected = prev.balance - curr.debit + curr.credit
        if abs(expected - curr.balance) > BALANCE_TOLERANCE:
            failures += 1
            if failures <= 5:
                result.add_warning(
                    "balance_mismatch",
                    f"Row {curr.row_index}: expected balance {expected}, "
                    f"statement says {curr.balance}.",
                    curr.row_index,
                )

    result.balance_checked = True
    result.balance_failures = failures

    ratio = failures / max(1, len(rows) - 1)
    if ratio > BALANCE_FAIL_THRESHOLD:
        result.add_error(
            "balance_check_failed",
            f"{failures} of {len(rows) - 1} rows do not reconcile against the "
            f"statement's own balance column ({ratio:.1%}). The file was most "
            f"likely parsed incorrectly, so it has been rejected rather than "
            f"producing wrong accounts.",
        )

    result.opening_balance = rows[0].balance - rows[0].credit + rows[0].debit
    result.closing_balance = rows[-1].balance
    return result


def deduplicate(result: ParseResult, scope: str = "") -> ParseResult:
    """Drop rows that are byte-identical *including* their position.

    Position is part of the identity on purpose - two genuine identical
    payments on the same day must both survive.
    """
    seen: set[str] = set()
    unique: List[Transaction] = []
    for txn in result.transactions:
        fp = txn.fingerprint(scope)
        if fp in seen:
            continue
        seen.add(fp)
        unique.append(txn)

    dropped = len(result.transactions) - len(unique)
    if dropped:
        result.add_warning("duplicates", f"Removed {dropped} duplicate row(s).")
    result.transactions = unique
    return result
