"""Default chart of accounts for an Indian SME (cash-basis, bank-only).

Deliberately small. A short list an accountant can scan beats a 200-line
textbook chart nobody maps correctly. Every head here is reachable from a
bank statement - there are no heads for stock, debtors or creditors, because
a bank statement cannot evidence them.

Codes follow the usual convention:
    1xxx assets   2xxx liabilities   3xxx equity   4xxx income   5xxx expenses
"""

from __future__ import annotations

from typing import Dict, List, Optional

from app.phase1.contracts import LedgerHead, Nature

# The bank account itself - one side of every posting in a bank-only pipeline.
BANK = "1010"
CASH = "1020"
CLOSING_STOCK = "1030"
# Where anything we cannot confidently classify goes. Never silently guessed.
SUSPENSE = "5999"
# Opening bank balance is carried as equity so the sheet is coherent.
OPENING_EQUITY = "3010"
# Profit for the period, transferred into equity.
CURRENT_PL = "3020"

_HEADS: List[LedgerHead] = [
    # ---- Assets -----------------------------------------------------------
    LedgerHead(BANK, "Bank Account", Nature.ASSET, "Current Assets", bs_section="Current Assets"),
    LedgerHead("1020", "Cash in Hand", Nature.ASSET, "Current Assets", bs_section="Current Assets"),
    LedgerHead("1110", "Fixed Assets", Nature.ASSET, "Fixed Assets", bs_section="Fixed Assets"),
    LedgerHead("1210", "Advance Tax / TDS Receivable", Nature.ASSET, "Current Assets", bs_section="Current Assets"),
    LedgerHead("1220", "Loans & Advances (Given)", Nature.ASSET, "Current Assets", bs_section="Current Assets"),
    LedgerHead("1230", "Investments & Deposits", Nature.ASSET, "Current Assets", bs_section="Current Assets"),

    # ---- Liabilities ------------------------------------------------------
    LedgerHead("2010", "Loan Account", Nature.LIABILITY, "Loans", bs_section="Current Liabilities"),
    LedgerHead("2020", "Credit Card Payable", Nature.LIABILITY, "Current Liabilities", bs_section="Current Liabilities"),
    LedgerHead("2110", "GST Payable", Nature.LIABILITY, "Duties & Taxes", bs_section="Current Liabilities"),
    LedgerHead("2120", "TDS Payable", Nature.LIABILITY, "Duties & Taxes", bs_section="Current Liabilities"),
    LedgerHead("2130", "Statutory Dues (PF/ESI/PT)", Nature.LIABILITY, "Duties & Taxes", bs_section="Current Liabilities"),

    # ---- Equity -----------------------------------------------------------
    LedgerHead(OPENING_EQUITY, "Opening Balance (Capital)", Nature.EQUITY, "Capital", bs_section="Capital Account"),
    LedgerHead(CURRENT_PL, "Profit for the Period", Nature.EQUITY, "Capital", bs_section="Capital Account"),
    LedgerHead("3030", "Capital Introduced", Nature.EQUITY, "Capital", bs_section="Capital Account"),
    LedgerHead("3040", "Drawings", Nature.EQUITY, "Capital", bs_section="Capital Account"),

    # ---- Income -----------------------------------------------------------
    LedgerHead("4010", "Sales / Receipts", Nature.INCOME, "Revenue", pl_section="Income"),
    LedgerHead("4020", "Interest Income", Nature.INCOME, "Other Income", pl_section="Other Income"),
    LedgerHead("4030", "Other Income", Nature.INCOME, "Other Income", pl_section="Other Income"),
    LedgerHead("4040", "Refunds Received", Nature.INCOME, "Other Income", pl_section="Other Income"),

    # ---- Expenses ---------------------------------------------------------
    LedgerHead("5010", "Rent", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5020", "Salaries & Wages", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5030", "Purchases / Supplier Payments", Nature.EXPENSE, "Direct Expenses", pl_section="Direct Expenses"),
    LedgerHead("5040", "Electricity & Utilities", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5050", "Telephone & Internet", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5060", "Bank Charges", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5070", "Interest Paid", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5080", "Travel & Conveyance", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5090", "Fuel", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5100", "Insurance", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5110", "Professional & Legal Fees", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5120", "Repairs & Maintenance", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5130", "Software & Subscriptions", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5140", "Advertising & Marketing", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5150", "Freight & Courier", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5160", "Office & General Expenses", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead("5170", "Taxes & Statutory Payments", Nature.EXPENSE, "Indirect Expenses", pl_section="Indirect Expenses"),
    LedgerHead(SUSPENSE, "Suspense (Needs Review)", Nature.EXPENSE, "Suspense", pl_section="Indirect Expenses"),
]

CHART: Dict[str, LedgerHead] = {h.code: h for h in _HEADS}


def get(code: Optional[str]) -> LedgerHead:
    """Look up a head, falling back to Suspense rather than raising.

    An unknown code is a data problem to surface in review, not a reason to
    fail a whole run.
    """
    if code and code in CHART:
        return CHART[code]
    return CHART[SUSPENSE]


def all_heads() -> List[LedgerHead]:
    return list(_HEADS)


def selectable_heads() -> List[Dict[str, str]]:
    """Chart as the review UI needs it - grouped, sorted, JSON-ready."""
    return [
        {
            "code": h.code,
            "name": h.name,
            "nature": h.nature.value,
            "group": h.group,
            "section": h.bs_section or h.pl_section or "",
        }
        for h in sorted(_HEADS, key=lambda x: x.code)
    ]
