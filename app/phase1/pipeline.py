"""Phase 1 orchestrator: bytes in, statements out. Zero LLM calls.

    run(file_bytes, filename)
        -> parse -> validate -> dedupe -> normalise -> rules -> journal
        -> ledgers -> trial balance -> P&L -> balance sheet

Every stage is deterministic, so the same file always produces the same
statements. That is the whole point of Phase 1: it makes the arithmetic
auditable and removes the model from everything except deciding what a
narration means.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from app.phase1 import coa, parsers
from app.phase1.contracts import (
    ZERO,
    GeneratedStatements,
    ParseResult,
    Source,
    Transaction,
    money,
)
from app.phase1.ledger import build_journal, build_ledgers, derive_opening_balance
from app.phase1.rules import RulesEngine
from app.phase1.statements import build_statements


@dataclass
class PipelineResult:
    """Everything one run produced, ready to serialise."""

    ok: bool
    transactions: List[Transaction] = field(default_factory=list)
    statements: Optional[GeneratedStatements] = None
    parse_result: Optional[ParseResult] = None
    coverage: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_ms: int = 0

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe view for the API layer."""
        s = self.statements
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
            "coverage": self.coverage,
            "opening_balance": str(s.opening_balance) if s else "0.00",
            "closing_balance": str(s.closing_balance) if s else "0.00",
            "net_profit": str(s.net_profit) if s else "0.00",
            "ties": s.ties if s else False,
            "trial_balance": s.trial_balance if s else [],
            "totals": {
                "debit": str(s.total_debits) if s else "0.00",
                "credit": str(s.total_credits) if s else "0.00",
            },
            "profit_and_loss": [
                {
                    "label": l.label,
                    "amount": str(l.amount),
                    "code": l.code,
                    "is_total": l.is_total,
                    "indent": l.indent,
                }
                for l in (s.pnl_lines if s else [])
            ],
            "balance_sheet": [
                {
                    "label": l.label,
                    "amount": str(l.amount),
                    "code": l.code,
                    "is_total": l.is_total,
                    "indent": l.indent,
                }
                for l in (s.balance_sheet_lines if s else [])
            ],
            "capital_account": [
                {
                    "label": l.label,
                    "amount": str(l.amount),
                    "code": l.code,
                    "is_total": l.is_total,
                    "indent": l.indent,
                }
                for l in (s.capital_account_lines if s else [])
            ],
            "client_info": (
                self.parse_result.statement_meta
                if self.parse_result and self.parse_result.statement_meta
                else (s.statement_meta if s and s.statement_meta else {})
            ),
            "transactions": [
                {
                    "row": t.row_index,
                    "date": t.txn_date.isoformat() if t.txn_date else "",
                    "narration": t.narration_raw,
                    "counterparty": t.counterparty_key,
                    "channel": t.channel.value,
                    "debit": str(t.debit),
                    "credit": str(t.credit),
                    "balance": str(t.balance) if t.balance is not None else "",
                    "ledger_code": t.ledger_code or "",
                    "ledger_name": coa.get(t.ledger_code).name,
                    "source": t.source.value,
                    "rule_id": t.rule_id or "",
                    "needs_review": t.source == Source.UNCLASSIFIED
                    or t.ledger_code == coa.SUSPENSE,
                }
                for t in self.transactions
            ],
        }


def _assemble(
    transactions: List[Transaction],
    opening_balance: Decimal,
    parsed: Optional[ParseResult],
    started: float,
    extra_warnings: Optional[List[str]] = None,
    custom_opening_capital: Optional[Decimal] = None,
) -> PipelineResult:
    """Journal -> ledgers -> statements. Shared by `run` and `rebuild`."""
    entries, journal_warnings = build_journal(transactions)
    ledgers = build_ledgers(entries, opening_bank=money(opening_balance))
    statements = build_statements(
        ledgers,
        opening_balance=money(opening_balance),
        custom_opening_capital=money(custom_opening_capital) if custom_opening_capital is not None else None,
        statement_meta=parsed.statement_meta if parsed else None,
    )

    warnings = list(extra_warnings or [])
    warnings.extend(journal_warnings)
    warnings.extend(statements.warnings)

    if parsed is not None and parsed.closing_balance is not None:
        if abs(statements.closing_balance - parsed.closing_balance) > Decimal("0.05"):
            warnings.append(
                f"Computed closing balance {statements.closing_balance} does not "
                f"match the statement's closing balance {parsed.closing_balance}."
            )

    return PipelineResult(
        ok=True,
        transactions=transactions,
        statements=statements,
        parse_result=parsed,
        coverage=RulesEngine.coverage(transactions),
        errors=[],
        warnings=warnings,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


def _apply_overrides(
    transactions: List[Transaction], overrides: Optional[Dict[int, str]]
) -> None:
    """A reviewer's choice always beats a rule."""
    if not overrides:
        return
    for txn in transactions:
        code = overrides.get(txn.row_index)
        if code and code in coa.CHART:
            txn.ledger_code = code
            txn.source = Source.HUMAN
            txn.rule_id = None


def rebuild(
    transaction_dicts: List[Dict[str, Any]],
    opening_balance: Decimal = ZERO,
    custom_opening_capital: Optional[Decimal] = None,
    overrides: Optional[Dict[int, str]] = None,
) -> PipelineResult:
    """Re-run the arithmetic from stored transactions, without re-parsing.

    Used when a reviewer reclassifies rows: parsing already happened, and
    re-doing it would risk a different result from the same file.
    """
    started = time.perf_counter()
    transactions = [Transaction.from_dict(d) for d in transaction_dicts]
    _apply_overrides(transactions, overrides)
    return _assemble(
        transactions,
        money(opening_balance),
        None,
        started,
        custom_opening_capital=custom_opening_capital,
    )


def run(
    content: bytes,
    filename: str,
    opening_balance: Optional[Decimal] = None,
    custom_opening_capital: Optional[Decimal] = None,
    overrides: Optional[Dict[int, str]] = None,
    scope: str = "",
) -> PipelineResult:
    """Full deterministic run.

    `overrides` maps `row_index -> ledger_code` and carries the reviewer's
    corrections back through a re-run. Human choices always beat rules.
    """
    started = time.perf_counter()

    parsed = parsers.parse(content, filename)
    if not parsed.transactions:
        return PipelineResult(
            ok=False,
            parse_result=parsed,
            errors=[i.message for i in parsed.errors] or ["No transactions found."],
            warnings=[i.message for i in parsed.warnings],
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    parsed = parsers.deduplicate(parsed, scope=scope)
    parsed = parsers.validate_running_balance(parsed)

    if parsed.errors:
        return PipelineResult(
            ok=False,
            transactions=parsed.transactions,
            parse_result=parsed,
            errors=[i.message for i in parsed.errors],
            warnings=[i.message for i in parsed.warnings],
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    transactions = RulesEngine().classify_all(parsed.transactions)
    _apply_overrides(transactions, overrides)

    if opening_balance is None:
        opening_balance = parsed.opening_balance
    if opening_balance is None:
        opening_balance = derive_opening_balance(transactions) or ZERO

    return _assemble(
        transactions,
        money(opening_balance),
        parsed,
        started,
        extra_warnings=[i.message for i in parsed.warnings],
        custom_opening_capital=custom_opening_capital,
    )

