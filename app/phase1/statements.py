"""Profit & Loss and Balance Sheet builders - pure code, no LLM.

This is a **cash-basis** presentation built from bank movement alone. It has
no debtors, creditors, stock, fixed-asset register or depreciation, because
a bank statement cannot evidence any of them. Every caller-facing surface
must say so; an accountant who reads it as an accrual balance sheet will be
misled.

Structure:
    P&L:  Income - Expenses = Net Profit
    BS :  Bank + Cash + other assets
          = Opening Capital + Net Profit - Drawings + liabilities
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Tuple

from app.phase1 import coa
from app.phase1.contracts import (
    ZERO,
    GeneratedStatements,
    LedgerAccount,
    Nature,
    StatementLine,
    money,
)
from app.phase1.ledger import account_totals, trial_balance

CASH_BASIS_NOTE = (
    "Cash-basis summary prepared from bank transactions only. It excludes "
    "receivables, payables, stock, fixed assets and depreciation, so it is "
    "not a statutory balance sheet."
)


def build_pnl(ledgers: Dict[str, LedgerAccount]) -> Tuple[List[StatementLine], Decimal]:
    """Income less expenses. Returns `(lines, net_profit)`."""
    totals = account_totals(ledgers)
    lines: List[StatementLine] = []

    income: List[Tuple[str, str, Decimal]] = []
    expenses: List[Tuple[str, str, Decimal]] = []
    for code, amount in totals.items():
        head = coa.get(code)
        if amount == ZERO:
            continue
        if head.nature == Nature.INCOME:
            income.append((code, head.name, amount))
        elif head.nature == Nature.EXPENSE:
            expenses.append((code, head.name, amount))

    income.sort(key=lambda r: r[2], reverse=True)
    expenses.sort(key=lambda r: r[2], reverse=True)

    total_income = money(sum((a for _, _, a in income), ZERO))
    total_expense = money(sum((a for _, _, a in expenses), ZERO))
    net_profit = money(total_income - total_expense)

    lines.append(StatementLine("INCOME", ZERO, is_total=True))
    for code, name, amount in income:
        lines.append(StatementLine(name, amount, code=code, indent=1))
    lines.append(StatementLine("Total Income", total_income, is_total=True))

    lines.append(StatementLine("EXPENSES", ZERO, is_total=True))
    for code, name, amount in expenses:
        lines.append(StatementLine(name, amount, code=code, indent=1))
    lines.append(StatementLine("Total Expenses", total_expense, is_total=True))

    label = "Net Profit" if net_profit >= ZERO else "Net Loss"
    lines.append(StatementLine(label, abs(net_profit), is_total=True))

    return lines, net_profit


def build_balance_sheet(
    ledgers: Dict[str, LedgerAccount],
    net_profit: Decimal,
) -> List[StatementLine]:
    """Assets against liabilities plus capital.

    Both sides are equal by construction - see the note in `ledger.py`.
    """
    totals = account_totals(ledgers)
    lines: List[StatementLine] = []

    assets: List[Tuple[str, str, Decimal]] = []
    liabilities: List[Tuple[str, str, Decimal]] = []
    equity: List[Tuple[str, str, Decimal]] = []

    for code, amount in totals.items():
        head = coa.get(code)
        if head.is_pl or amount == ZERO:
            continue
        if head.nature == Nature.ASSET:
            assets.append((code, head.name, amount))
        elif head.nature == Nature.LIABILITY:
            liabilities.append((code, head.name, amount))
        elif head.nature == Nature.EQUITY:
            # Drawings reduce capital, so carry it negative.
            if code == "3040":
                equity.append((code, "Less: Drawings", -abs(amount)))
            else:
                equity.append((code, head.name, amount))

    equity.append((coa.CURRENT_PL, coa.get(coa.CURRENT_PL).name, money(net_profit)))

    assets.sort(key=lambda r: r[2], reverse=True)
    liabilities.sort(key=lambda r: r[2], reverse=True)

    total_assets = money(sum((a for _, _, a in assets), ZERO))
    total_liabilities = money(sum((a for _, _, a in liabilities), ZERO))
    total_equity = money(sum((a for _, _, a in equity), ZERO))

    lines.append(StatementLine("ASSETS", ZERO, is_total=True))
    for code, name, amount in assets:
        lines.append(StatementLine(name, amount, code=code, indent=1))
    lines.append(StatementLine("Total Assets", total_assets, is_total=True))

    lines.append(StatementLine("LIABILITIES", ZERO, is_total=True))
    if liabilities:
        for code, name, amount in liabilities:
            lines.append(StatementLine(name, amount, code=code, indent=1))
    else:
        lines.append(StatementLine("None", ZERO, indent=1))
    lines.append(StatementLine("Total Liabilities", total_liabilities, is_total=True))

    lines.append(StatementLine("CAPITAL ACCOUNT", ZERO, is_total=True))
    for code, name, amount in equity:
        lines.append(StatementLine(name, amount, code=code, indent=1))
    lines.append(StatementLine("Total Capital", total_equity, is_total=True))

    lines.append(
        StatementLine(
            "Total Liabilities + Capital", money(total_liabilities + total_equity), is_total=True
        )
    )
    return lines


def build_capital_account(
    ledgers: Dict[str, LedgerAccount],
    net_profit: Decimal,
    opening_balance: Decimal = ZERO,
    custom_opening_capital: Decimal | None = None,
) -> List[StatementLine]:
    """Capital Account schedule (as presented in Indian accounting / CAP statement):
    Opening Capital Balance
    + Fresh Capital Introduced (if any)
    + Net Profit / (- Net Loss)
    - Drawings (3040)
    = Closing Capital Balance
    """
    totals = account_totals(ledgers)
    lines: List[StatementLine] = []

    opening_cap = custom_opening_capital if custom_opening_capital is not None else opening_balance
    lines.append(StatementLine("Opening Capital Balance", money(opening_cap)))

    # Capital Introduced (Head 3030)
    cap_introduced = money(totals.get("3030", ZERO))
    if cap_introduced > ZERO:
        lines.append(StatementLine("Add: Capital Introduced", cap_introduced, indent=1))

    if net_profit >= ZERO:
        lines.append(StatementLine("Add: Net Profit for the Year", net_profit, indent=1))
    else:
        lines.append(StatementLine("Less: Net Loss for the Year", -abs(net_profit), indent=1))

    drawings = abs(money(totals.get("3040", ZERO)))
    if drawings > ZERO:
        lines.append(StatementLine("Less: Drawings / Personal Withdrawals", -drawings, indent=1))

    closing_capital = money(
        opening_cap + cap_introduced + net_profit - drawings
    )
    lines.append(StatementLine("Closing Capital Balance (to Balance Sheet)", closing_capital, is_total=True))

    return lines


def build_statements(
    ledgers: Dict[str, LedgerAccount],
    opening_balance: Decimal = ZERO,
    custom_opening_capital: Decimal | None = None,
    statement_meta: Dict[str, str] | None = None,
) -> GeneratedStatements:
    """Run the full deterministic statement build and self-check it."""
    tb_rows, total_debits, total_credits = trial_balance(ledgers)
    pnl_lines, net_profit = build_pnl(ledgers)
    bs_lines = build_balance_sheet(ledgers, net_profit)
    cap_lines = build_capital_account(
        ledgers,
        net_profit,
        opening_balance=money(opening_balance),
        custom_opening_capital=money(custom_opening_capital) if custom_opening_capital is not None else None,
    )

    totals = account_totals(ledgers)
    closing = money(totals.get(coa.BANK, ZERO))

    result = GeneratedStatements(
        trial_balance=[
            {
                "code": r["code"],
                "name": r["name"],
                "debit": str(r["debit"]),
                "credit": str(r["credit"]),
            }
            for r in tb_rows
        ],
        total_debits=total_debits,
        total_credits=total_credits,
        pnl_lines=pnl_lines,
        balance_sheet_lines=bs_lines,
        capital_account_lines=cap_lines,
        statement_meta=statement_meta or {},
        net_profit=net_profit,
        opening_balance=money(opening_balance),
        closing_balance=closing,
    )

    if not result.ties:
        # Should be unreachable; if it fires, the engine has a bug.
        result.warnings.append(
            f"TRIAL BALANCE DOES NOT TIE: debits {total_debits} vs credits "
            f"{total_credits}. Do not rely on these figures."
        )

    suspense = money(totals.get(coa.SUSPENSE, ZERO))
    if suspense != ZERO:
        result.warnings.append(
            f"{coa.get(coa.SUSPENSE).name}: {suspense} needs review before "
            f"these statements mean anything."
        )

    return result

