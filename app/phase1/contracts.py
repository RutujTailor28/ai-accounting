"""Canonical data contract for the Phase 1 deterministic pipeline.

Everything in `app/phase1` speaks these types. Freezing them is what lets the
parser, the ledger engine and the statement builder be developed and tested
independently.

Money is `Decimal` everywhere - never `float`. Rupee amounts summed as floats
drift by paise over a few thousand rows and a trial balance that is out by
0.01 is indistinguishable from one that is out by 100000.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

TWO_PLACES = Decimal("0.01")
ZERO = Decimal("0.00")


def money(value: Any) -> Decimal:
    """Coerce anything to a 2dp Decimal. Never raises - returns 0.00 instead.

    Goes through `str()` so we never inherit binary float error
    (`Decimal(0.1)` is 0.1000000000000000055511151231257827,
    `Decimal("0.1")` is exactly 0.1).
    """
    if value is None or value == "":
        return ZERO
    try:
        if isinstance(value, Decimal):
            d = value
        elif isinstance(value, float):
            d = Decimal(str(value))
        elif isinstance(value, int):
            d = Decimal(value)
        else:
            cleaned = (
                str(value)
                .replace(",", "")
                .replace("₹", "")
                .replace("Rs.", "")
                .replace("Rs", "")
                .replace("INR", "")
                .strip()
            )
            if cleaned in ("", "-", "--", "nan", "NaN", "None"):
                return ZERO
            # Trailing/leading CR-DR markers, and (1234) for negatives.
            negative = False
            if cleaned.startswith("(") and cleaned.endswith(")"):
                negative, cleaned = True, cleaned[1:-1]
            upper = cleaned.upper()
            for marker in ("DR", "CR", "D", "C"):
                if upper.endswith(marker):
                    cleaned = cleaned[: -len(marker)].strip()
                    break
            d = Decimal(cleaned or "0")
            if negative:
                d = -d
    except Exception:
        return ZERO
    return d.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


class Channel(str, Enum):
    """How the money moved. Derived from the narration, not the amount."""

    UPI = "UPI"
    NEFT = "NEFT"
    IMPS = "IMPS"
    RTGS = "RTGS"
    CHEQUE = "CHQ"
    ATM = "ATM"
    POS = "POS"
    CHARGES = "CHARGES"
    INTEREST = "INT"
    CASH = "CASH"
    TRANSFER = "TRF"
    OTHER = "OTHER"


class Nature(str, Enum):
    """Which of the five account natures a ledger head belongs to."""

    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    INCOME = "income"
    EXPENSE = "expense"


class Source(str, Enum):
    """How a row got its ledger. Drives what the reviewer sees first."""

    RULE = "rule"
    HUMAN = "human"
    UNCLASSIFIED = "unclassified"


@dataclass
class LedgerHead:
    """One line in the chart of accounts."""

    code: str
    name: str
    nature: Nature
    group: str = ""
    # Where it appears. Exactly one of these is set for a real head.
    bs_section: Optional[str] = None
    pl_section: Optional[str] = None

    @property
    def is_pl(self) -> bool:
        return self.nature in (Nature.INCOME, Nature.EXPENSE)

    @property
    def normal_side(self) -> str:
        """The side that increases this account."""
        return "debit" if self.nature in (Nature.ASSET, Nature.EXPENSE) else "credit"


@dataclass
class Transaction:
    """One row of a bank statement, in canonical form.

    `debit` = money out of the bank. `credit` = money into the bank.
    Exactly one of them is non-zero on a well-formed row.
    """

    txn_date: Optional[date]
    narration_raw: str
    debit: Decimal = ZERO
    credit: Decimal = ZERO
    balance: Optional[Decimal] = None

    # Filled by the normaliser
    narration_clean: str = ""
    counterparty_key: str = ""
    channel: Channel = Channel.OTHER

    # Filled by the rules engine / reviewer
    ledger_code: Optional[str] = None
    source: Source = Source.UNCLASSIFIED
    rule_id: Optional[str] = None

    # Provenance
    row_index: int = 0
    source_document: str = ""
    ref_no: str = ""

    @property
    def amount(self) -> Decimal:
        """Absolute movement, whichever direction."""
        return self.debit if self.debit > ZERO else self.credit

    @property
    def direction(self) -> str:
        return "debit" if self.debit > ZERO else "credit"

    @property
    def is_valid(self) -> bool:
        """A row must move money in exactly one direction."""
        return (self.debit > ZERO) != (self.credit > ZERO)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for storage. Money is stored as a string so it survives
        a JSON round trip without becoming a float."""
        return {
            "txn_date": self.txn_date.isoformat() if self.txn_date else None,
            "narration_raw": self.narration_raw,
            "debit": str(self.debit),
            "credit": str(self.credit),
            "balance": str(self.balance) if self.balance is not None else None,
            "narration_clean": self.narration_clean,
            "counterparty_key": self.counterparty_key,
            "channel": self.channel.value,
            "ledger_code": self.ledger_code,
            "source": self.source.value,
            "rule_id": self.rule_id,
            "row_index": self.row_index,
            "source_document": self.source_document,
            "ref_no": self.ref_no,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transaction":
        from datetime import date as _date

        raw_date = data.get("txn_date")
        parsed_date: Optional[_date] = None
        if raw_date:
            try:
                parsed_date = _date.fromisoformat(raw_date)
            except Exception:
                parsed_date = None

        balance = data.get("balance")
        return cls(
            txn_date=parsed_date,
            narration_raw=data.get("narration_raw", ""),
            debit=money(data.get("debit")),
            credit=money(data.get("credit")),
            balance=money(balance) if balance is not None else None,
            narration_clean=data.get("narration_clean", ""),
            counterparty_key=data.get("counterparty_key", ""),
            channel=Channel(data.get("channel", "OTHER")),
            ledger_code=data.get("ledger_code"),
            source=Source(data.get("source", "unclassified")),
            rule_id=data.get("rule_id"),
            row_index=int(data.get("row_index", 0)),
            source_document=data.get("source_document", ""),
            ref_no=data.get("ref_no", ""),
        )

    def fingerprint(self, scope: str = "") -> str:
        """Stable identity for de-duplication.

        `row_index` is deliberately part of the hash. Without it two genuine
        identical payments to the same vendor on the same day collapse into
        one row, silently understating expenses and breaking the running
        balance.
        """
        parts = [
            scope,
            self.source_document,
            str(self.txn_date or ""),
            str(self.debit),
            str(self.credit),
            self.narration_raw.strip().lower(),
            str(self.row_index),
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


@dataclass
class ParseIssue:
    """Something the parser could not do cleanly."""

    severity: str  # "error" | "warning"
    code: str
    message: str
    row_index: Optional[int] = None


@dataclass
class ParseResult:
    """What a bank statement parser returns."""

    transactions: List[Transaction] = field(default_factory=list)
    issues: List[ParseIssue] = field(default_factory=list)
    bank_hint: str = ""
    source_document: str = ""
    # Populated by the running-balance check
    balance_checked: bool = False
    balance_failures: int = 0
    opening_balance: Optional[Decimal] = None
    closing_balance: Optional[Decimal] = None
    statement_meta: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.transactions) and not any(
            i.severity == "error" for i in self.issues
        )

    @property
    def errors(self) -> List[ParseIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ParseIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def add_error(self, code: str, message: str, row_index: Optional[int] = None) -> None:
        self.issues.append(ParseIssue("error", code, message, row_index))

    def add_warning(self, code: str, message: str, row_index: Optional[int] = None) -> None:
        self.issues.append(ParseIssue("warning", code, message, row_index))


@dataclass
class JournalEntry:
    """One balanced double-entry posting."""

    entry_date: Optional[date]
    debit_code: str
    credit_code: str
    amount: Decimal
    narration: str
    txn_row_index: int = 0

    def __post_init__(self) -> None:
        self.amount = money(self.amount)


@dataclass
class LedgerAccount:
    """Running position of one ledger head."""

    code: str
    name: str
    nature: Nature
    debits: Decimal = ZERO
    credits: Decimal = ZERO
    opening: Decimal = ZERO

    @property
    def balance(self) -> Decimal:
        """Signed on the account's normal side (always non-negative for
        well-formed data)."""
        raw = self.opening + self.debits - self.credits
        if self.normal_side == "credit":
            raw = -(self.opening) + self.credits - self.debits
        return money(raw)

    @property
    def normal_side(self) -> str:
        return "debit" if self.nature in (Nature.ASSET, Nature.EXPENSE) else "credit"


@dataclass
class StatementLine:
    """One printed line of a P&L or balance sheet."""

    label: str
    amount: Decimal
    code: str = ""
    is_total: bool = False
    indent: int = 0


@dataclass
class GeneratedStatements:
    """The full deterministic output for one run."""

    trial_balance: List[Dict[str, Any]] = field(default_factory=list)
    total_debits: Decimal = ZERO
    total_credits: Decimal = ZERO
    pnl_lines: List[StatementLine] = field(default_factory=list)
    balance_sheet_lines: List[StatementLine] = field(default_factory=list)
    capital_account_lines: List[StatementLine] = field(default_factory=list)
    statement_meta: Dict[str, str] = field(default_factory=dict)
    net_profit: Decimal = ZERO
    opening_balance: Decimal = ZERO
    closing_balance: Decimal = ZERO
    warnings: List[str] = field(default_factory=list)

    @property
    def ties(self) -> bool:
        return self.total_debits == self.total_credits
