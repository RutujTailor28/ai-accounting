"""Ledger engine - pure arithmetic, Decimal only, no LLM anywhere.

Bank transaction -> journal entry -> ledger accounts -> trial balance.

Because every posting is constructed balanced (bank on one side, the
classified head on the other), the trial balance always ties. That check
therefore catches *coding* bugs, not accounting mistakes: rent misposted as
salary ties perfectly. Do not read a tying trial balance as evidence the
classification is right.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from app.phase1 import coa
from app.phase1.contracts import (
    ZERO,
    JournalEntry,
    LedgerAccount,
    Nature,
    Transaction,
    money,
)


def build_journal(
    transactions: List[Transaction],
    bank_code: str = coa.BANK,
) -> Tuple[List[JournalEntry], List[str]]:
    """Turn classified bank rows into balanced double entries.

    Money out of the bank  -> Dr <head>, Cr Bank
    Money into the bank    -> Dr Bank,   Cr <head>
    """
    entries: List[JournalEntry] = []
    warnings: List[str] = []

    for txn in transactions:
        if not txn.is_valid:
            warnings.append(
                f"Row {txn.row_index}: skipped - must have exactly one of "
                f"debit/credit (got debit={txn.debit}, credit={txn.credit})"
            )
            continue

        amount = money(txn.amount)
        if amount <= ZERO:
            warnings.append(f"Row {txn.row_index}: skipped - zero or negative amount")
            continue

        head_code = txn.ledger_code or coa.SUSPENSE
        if txn.direction == "debit":
            debit_code, credit_code = head_code, bank_code
        else:
            debit_code, credit_code = bank_code, head_code

        entries.append(
            JournalEntry(
                entry_date=txn.txn_date,
                debit_code=debit_code,
                credit_code=credit_code,
                amount=amount,
                narration=txn.narration_raw[:200],
                txn_row_index=txn.row_index,
            )
        )

    return entries, warnings


def build_ledgers(
    entries: List[JournalEntry],
    opening_bank: Decimal = ZERO,
    bank_code: str = coa.BANK,
    opening_equity_code: str = coa.OPENING_EQUITY,
) -> Dict[str, LedgerAccount]:
    """Post every entry into ledger accounts.

    The opening bank balance is seeded as a matching pair - Dr Bank,
    Cr Opening Capital - so the books start balanced. Without this the
    closing bank figure is right but nothing explains where it came from.
    """
    ledgers: Dict[str, LedgerAccount] = {}

    def account_for(code: str) -> LedgerAccount:
        if code not in ledgers:
            head = coa.get(code)
            ledgers[code] = LedgerAccount(
                code=head.code, name=head.name, nature=head.nature
            )
        return ledgers[code]

    opening_bank = money(opening_bank)
    if opening_bank != ZERO:
        if opening_bank > ZERO:
            account_for(bank_code).debits += opening_bank
            account_for(opening_equity_code).credits += opening_bank
        else:
            # Overdrawn at the start: bank is a credit balance.
            account_for(bank_code).credits += -opening_bank
            account_for(opening_equity_code).debits += -opening_bank

    for entry in entries:
        account_for(entry.debit_code).debits += entry.amount
        account_for(entry.credit_code).credits += entry.amount

    for acc in ledgers.values():
        acc.debits = money(acc.debits)
        acc.credits = money(acc.credits)

    return ledgers


def trial_balance(ledgers: Dict[str, LedgerAccount]) -> Tuple[List[Dict], Decimal, Decimal]:
    """Net each account onto one side and total both columns.

    Returns `(rows, total_debits, total_credits)`. The totals must be equal;
    the caller should treat inequality as a hard failure.
    """
    rows: List[Dict] = []
    total_debit = ZERO
    total_credit = ZERO

    for code in sorted(ledgers):
        acc = ledgers[code]
        net = money(acc.debits - acc.credits)
        if net == ZERO and acc.debits == ZERO and acc.credits == ZERO:
            continue

        debit_col = net if net > ZERO else ZERO
        credit_col = -net if net < ZERO else ZERO
        total_debit += debit_col
        total_credit += credit_col

        rows.append(
            {
                "code": acc.code,
                "name": acc.name,
                "nature": acc.nature.value,
                "debit": debit_col,
                "credit": credit_col,
            }
        )

    return rows, money(total_debit), money(total_credit)


def account_totals(ledgers: Dict[str, LedgerAccount]) -> Dict[str, Decimal]:
    """Net movement per account, signed on its normal side.

    Income and liabilities come back positive when credited, which is what
    the statement builder wants.
    """
    totals: Dict[str, Decimal] = {}
    for code, acc in ledgers.items():
        head = coa.get(code)
        net = acc.debits - acc.credits
        if head.nature in (Nature.INCOME, Nature.LIABILITY, Nature.EQUITY):
            net = -net
        totals[code] = money(net)
    return totals


def derive_opening_balance(transactions: List[Transaction]) -> Optional[Decimal]:
    """Recover the opening bank balance from the statement's own running
    balance column.

    The first row's closing balance minus that row's own effect is the
    balance before it. Returns None when the statement carries no balance
    column, in which case the caller must ask the user for it.
    """
    for txn in transactions:
        if txn.balance is None:
            continue
        effect = txn.credit - txn.debit
        return money(txn.balance - effect)
    return None
