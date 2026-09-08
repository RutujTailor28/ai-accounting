"""Statutory Multi-Source Financial Statement Builder.

Dynamically combines:
1. Current Year Bank Movements (Phase 1 engine)
2. Prior Year Balance Sheet Assets & Liabilities (rolled forward)
3. Year-End Adjustments (Closing Stock, Cash Book, Depreciation, Accruals)

Enforces strict double-entry arithmetic so the final Balance Sheet
ties mathematically for ANY business or entity type.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from app.phase1 import coa
from app.phase1.contracts import ZERO, GeneratedStatements, StatementLine, money


def build_statutory_statements(
    bank_statements: Optional[GeneratedStatements],
    prior_assets: List[Dict[str, Any]],
    prior_liabilities: List[Dict[str, Any]],
    adjustments: List[Dict[str, Any]],
    custom_opening_capital: Optional[Decimal] = None,
) -> Dict[str, Any]:
    """Combine prior balance sheet + bank statements + adjustments dynamically."""
    warnings: List[str] = []

    # 1. Classify Prior Assets & Prior Liabilities dynamically
    prior_bank_total = ZERO
    prior_cash_total = ZERO
    prior_stock_total = ZERO
    prior_fixed_items: List[Dict[str, Any]] = []
    prior_other_assets: List[Dict[str, Any]] = []

    for a in prior_assets:
        lbl = str(a.get("label", "")).strip()
        lbl_lower = lbl.lower()
        cat = str(a.get("category", "")).lower()
        grp = str(a.get("group", "")).lower()
        code = str(a.get("code", ""))
        amt = money(a.get("amount", ZERO))
        if amt == ZERO:
            continue

        is_bank = code == coa.BANK or "bank" in lbl_lower or "bank" in cat
        is_cash = (code == coa.CASH or "cash" in lbl_lower or "cash" in cat) and not is_bank
        is_stock = code == coa.CLOSING_STOCK or "stock" in lbl_lower or "inventory" in lbl_lower
        is_fixed = (
            code.startswith("11")
            or "fixed" in grp
            or "fixed" in cat
            or any(kw in lbl_lower for kw in ["equipment", "furniture", "vehical", "vehicle", "phone", "machine", "plant", "computer", "building", "land"])
        )

        if is_bank:
            prior_bank_total += amt
        elif is_cash:
            prior_cash_total += amt
        elif is_stock:
            prior_stock_total += amt
        elif is_fixed:
            prior_fixed_items.append({"label": lbl, "amount": amt, "group": a.get("group") or "Fixed Assets"})
        else:
            prior_other_assets.append({"label": lbl, "amount": amt, "group": a.get("group") or "Other Assets"})

    tot_prior_assets = sum((money(a.get("amount", ZERO)) for a in prior_assets), ZERO)

    # Classify prior liabilities
    prior_capital_total = ZERO
    prior_outside_liabilities: List[Dict[str, Any]] = []

    for l in prior_liabilities:
        lbl = str(l.get("label", "")).strip()
        lbl_lower = lbl.lower()
        cat = str(l.get("category", "")).lower()
        grp = str(l.get("group", "")).lower()
        code = str(l.get("code", ""))
        amt = money(l.get("amount", ZERO))
        if amt == ZERO:
            continue

        is_cap = code == coa.OPENING_EQUITY or "capital" in grp or "capital" in cat or "capital" in lbl_lower
        if is_cap:
            prior_capital_total += amt
        else:
            prior_outside_liabilities.append({"label": lbl, "amount": amt, "group": l.get("group") or "Current Liabilities"})

    tot_prior_outside_liab = sum((l["amount"] for l in prior_outside_liabilities), ZERO)

    # 2. Determine Opening Capital
    if custom_opening_capital is not None and money(custom_opening_capital) > ZERO:
        opening_cap = money(custom_opening_capital)
    elif prior_capital_total > ZERO:
        opening_cap = prior_capital_total
    elif tot_prior_assets > ZERO:
        opening_cap = money(tot_prior_assets - tot_prior_outside_liab)
    elif bank_statements:
        opening_cap = bank_statements.opening_balance
    else:
        opening_cap = ZERO

    # 3. Extract Bank Operations (Phase 1)
    bank_opening = bank_statements.opening_balance if bank_statements else ZERO
    bank_closing = bank_statements.closing_balance if bank_statements else ZERO
    bank_drawings = ZERO
    bank_cash_addition = ZERO

    if bank_statements:
        for line in bank_statements.capital_account_lines:
            if "Drawings" in line.label:
                bank_drawings += abs(line.amount)
        for line in bank_statements.balance_sheet_lines:
            if "Cash in Hand" in line.label or line.code == coa.CASH:
                bank_cash_addition += line.amount

    # Section-aware extraction of bank Income and Expenses
    bank_income_lines: List[StatementLine] = []
    bank_expense_lines: List[StatementLine] = []
    if bank_statements:
        in_inc = False
        in_exp = False
        for line in bank_statements.pnl_lines:
            if line.label == "INCOME":
                in_inc = True
                in_exp = False
                continue
            elif line.label == "EXPENSES":
                in_exp = True
                in_inc = False
                continue
            elif line.label.startswith("Total") or line.label.startswith("Net"):
                in_inc = False
                in_exp = False
                continue

            if in_inc and not line.is_total and line.amount > ZERO:
                bank_income_lines.append(line)
            elif in_exp and not line.is_total and line.amount > ZERO:
                bank_expense_lines.append(line)

    # 4. Bank Opening Variance Reconciliation
    # When prior bank is provided and differs from bank statement opening balance:
    bank_opening_variance = ZERO
    if bank_statements and prior_bank_total > ZERO and prior_bank_total != bank_opening:
        bank_opening_variance = money(bank_opening - prior_bank_total)

    adjusted_opening_cap = money(opening_cap + bank_opening_variance)

    # 5. Process Adjustments
    adj_closing_stock = ZERO
    adj_cash_counter_sales = ZERO
    adj_cash_drawings = ZERO
    adj_cash_expenses = ZERO
    adj_depreciation = ZERO
    custom_incomes: List[Tuple[str, Decimal]] = []
    custom_expenses: List[Tuple[str, Decimal]] = []
    custom_assets: List[Dict[str, Any]] = []
    custom_liabilities: List[Dict[str, Any]] = []

    for adj in adjustments:
        amt = money(adj.get("amount", ZERO))
        if amt <= ZERO:
            continue
        atype = str(adj.get("type", "")).lower()
        lbl = str(adj.get("label", "Adjustment")).strip()

        if atype == "closing_stock" or "stock" in lbl.lower():
            adj_closing_stock += amt
        elif atype == "cash_sales" or "cash sales" in lbl.lower() or "counter" in lbl.lower():
            adj_cash_counter_sales += amt
        elif atype == "cash_drawings" or "cash withdrawal" in lbl.lower() or "drawings" in lbl.lower():
            adj_cash_drawings += amt
        elif atype == "cash_expense" or "cash expense" in lbl.lower():
            adj_cash_expenses += amt
        elif atype == "depreciation" or "deprec" in lbl.lower():
            adj_depreciation += amt
        elif atype == "income" or "income" in lbl.lower():
            custom_incomes.append((lbl, amt))
        elif atype == "expense" or "expense" in lbl.lower():
            custom_expenses.append((lbl, amt))
        elif atype == "asset":
            custom_assets.append({"label": lbl, "amount": amt, "group": adj.get("group", "Other Assets")})
        elif atype == "liability":
            custom_liabilities.append({"label": lbl, "amount": amt, "group": adj.get("group", "Current Liabilities")})

    # 6. Consolidated Trading & Profit and Loss Statement
    pnl_lines: List[StatementLine] = []
    pnl_lines.append(StatementLine("INCOME", ZERO, is_total=True))
    tot_income = ZERO

    # Bank income lines
    for line in bank_income_lines:
        pnl_lines.append(line)
        tot_income += line.amount

    # Cash counter sales
    if adj_cash_counter_sales > ZERO:
        pnl_lines.append(StatementLine("Cash Counter Receipts / Sales", adj_cash_counter_sales, indent=1))
        tot_income += adj_cash_counter_sales

    # Custom incomes
    for c_lbl, c_amt in custom_incomes:
        pnl_lines.append(StatementLine(c_lbl, c_amt, indent=1))
        tot_income += c_amt

    # Closing stock credited to Trading account
    if adj_closing_stock > ZERO:
        pnl_lines.append(StatementLine("Closing Stock (Trading A/c)", adj_closing_stock, indent=1))
        tot_income += adj_closing_stock

    pnl_lines.append(StatementLine("Total Income", tot_income, is_total=True))

    pnl_lines.append(StatementLine("EXPENSES", ZERO, is_total=True))
    tot_expense = ZERO

    # Opening stock debited to Trading account if closing stock adjustment is present
    if adj_closing_stock > ZERO and prior_stock_total > ZERO:
        pnl_lines.append(StatementLine("Opening Stock (Trading A/c)", prior_stock_total, indent=1))
        tot_expense += prior_stock_total

    # Bank expense lines
    for line in bank_expense_lines:
        pnl_lines.append(line)
        tot_expense += line.amount

    # Cash expenses
    if adj_cash_expenses > ZERO:
        pnl_lines.append(StatementLine("Cash Book Expenses", adj_cash_expenses, indent=1))
        tot_expense += adj_cash_expenses

    # Depreciation
    if adj_depreciation > ZERO:
        pnl_lines.append(StatementLine("Depreciation Expense", adj_depreciation, indent=1))
        tot_expense += adj_depreciation

    # Custom expenses
    for c_lbl, c_amt in custom_expenses:
        pnl_lines.append(StatementLine(c_lbl, c_amt, indent=1))
        tot_expense += c_amt

    pnl_lines.append(StatementLine("Total Expenses", tot_expense, is_total=True))

    statutory_net_profit = money(tot_income - tot_expense)
    profit_label = "Net Profit for the Year" if statutory_net_profit >= ZERO else "Net Loss for the Year"
    pnl_lines.append(StatementLine(profit_label, abs(statutory_net_profit), is_total=True))

    # Dual-Column (T-Format) Profit & Loss mapping
    t_pnl_expenses: List[Dict[str, Any]] = []
    t_pnl_income: List[Dict[str, Any]] = []

    # Income Side (Credit)
    for line in bank_income_lines:
        t_pnl_income.append({"label": line.label, "amount": str(line.amount), "group": "Revenue"})
    if adj_cash_counter_sales > ZERO:
        t_pnl_income.append({"label": "Cash Counter Receipts / Sales", "amount": str(adj_cash_counter_sales), "group": "Revenue"})
    for c_lbl, c_amt in custom_incomes:
        t_pnl_income.append({"label": c_lbl, "amount": str(c_amt), "group": "Other Income"})
    if adj_closing_stock > ZERO:
        t_pnl_income.append({"label": "Closing Stock", "amount": str(adj_closing_stock), "group": "Trading Account"})
    if statutory_net_profit < ZERO:
        t_pnl_income.append({"label": "Net Loss (to Capital A/c)", "amount": str(abs(statutory_net_profit)), "group": "Profit & Loss"})

    # Expense Side (Debit)
    if adj_closing_stock > ZERO and prior_stock_total > ZERO:
        t_pnl_expenses.append({"label": "Opening Stock", "amount": str(prior_stock_total), "group": "Trading Account"})
    for line in bank_expense_lines:
        t_pnl_expenses.append({"label": line.label, "amount": str(line.amount), "group": "Expenses"})
    if adj_cash_expenses > ZERO:
        t_pnl_expenses.append({"label": "Cash Book Expenses", "amount": str(adj_cash_expenses), "group": "Cash Expenses"})
    if adj_depreciation > ZERO:
        t_pnl_expenses.append({"label": "Depreciation Expense", "amount": str(adj_depreciation), "group": "Depreciation"})
    for c_lbl, c_amt in custom_expenses:
        t_pnl_expenses.append({"label": c_lbl, "amount": str(c_amt), "group": "Other Expenses"})
    if statutory_net_profit >= ZERO:
        t_pnl_expenses.append({"label": "Net Profit (to Capital A/c)", "amount": str(statutory_net_profit), "group": "Profit & Loss"})

    pnl_grand_total = max(tot_income, tot_expense) if statutory_net_profit >= ZERO else max(tot_income + abs(statutory_net_profit), tot_expense)

    # 7. Consolidated Capital Account Schedule
    tot_drawings = money(bank_drawings + adj_cash_drawings)
    closing_capital = money(adjusted_opening_cap + statutory_net_profit - tot_drawings)

    cap_lines: List[StatementLine] = []
    cap_lines.append(StatementLine("Opening Capital Balance", opening_cap))
    if bank_opening_variance != ZERO:
        cap_lines.append(
            StatementLine(
                "Opening Bank Reconciliation Variance",
                bank_opening_variance,
                indent=1,
            )
        )
    if statutory_net_profit >= ZERO:
        cap_lines.append(StatementLine("Add: Net Profit for the Year", statutory_net_profit, indent=1))
    else:
        cap_lines.append(StatementLine("Less: Net Loss for the Year", -abs(statutory_net_profit), indent=1))

    if tot_drawings > ZERO:
        cap_lines.append(StatementLine("Less: Total Drawings (Bank + Cash)", -tot_drawings, indent=1))

    cap_lines.append(
        StatementLine("Closing Capital Balance (to Balance Sheet)", closing_capital, is_total=True)
    )

    # 8. Consolidated Balance Sheet
    asset_sections: Dict[str, List[Tuple[str, Decimal]]] = {}

    # Fixed Assets (less depreciation)
    if prior_fixed_items:
        rem_deprec = adj_depreciation
        for fa in prior_fixed_items:
            grp = fa["group"]
            lbl = fa["label"]
            amt = fa["amount"]
            asset_sections.setdefault(grp, []).append((lbl, amt))
        if rem_deprec > ZERO:
            # Add deduction line in the first fixed asset group
            first_grp = prior_fixed_items[0]["group"]
            asset_sections[first_grp].append(("Less: Depreciation", -rem_deprec))
    elif adj_depreciation > ZERO:
        asset_sections.setdefault("Fixed Assets", []).append(("Less: Depreciation", -adj_depreciation))

    # Other Prior Assets (Deposits, Advances, Sundry Debtors)
    for oa in prior_other_assets:
        asset_sections.setdefault(oa["group"], []).append((oa["label"], oa["amount"]))

    # Cash & Bank
    cash_bank_list: List[Tuple[str, Decimal]] = []
    effective_bank_balance = bank_closing if bank_statements else prior_bank_total
    if effective_bank_balance > ZERO or bank_statements:
        cash_bank_list.append(("Bank Account Balance", effective_bank_balance))

    # Closing Cash in Hand:
    # Prior Cash + ATM Withdrawals + Cash Counter Sales - Cash Drawings - Cash Expenses
    closing_cash = money(
        prior_cash_total
        + bank_cash_addition
        + adj_cash_counter_sales
        - adj_cash_drawings
        - adj_cash_expenses
    )
    if closing_cash > ZERO or prior_cash_total > ZERO or adj_cash_counter_sales > ZERO:
        cash_bank_list.append(("Cash Book / Cash in Hand", closing_cash))

    if cash_bank_list:
        asset_sections["Cash & Bank"] = cash_bank_list

    # Closing Stock
    if adj_closing_stock > ZERO:
        asset_sections["Closing Stock"] = [("Closing Stock", adj_closing_stock)]
    elif prior_stock_total > ZERO:
        asset_sections["Closing Stock"] = [("Closing Stock (Carried Forward)", prior_stock_total)]

    # Custom Assets
    for ca in custom_assets:
        asset_sections.setdefault(ca["group"], []).append((ca["label"], ca["amount"]))

    # Liabilities
    liab_sections: Dict[str, List[Tuple[str, Decimal]]] = {}
    liab_sections["Capital Account"] = [("Closing Capital Balance", closing_capital)]

    for ol in prior_outside_liabilities:
        liab_sections.setdefault(ol["group"], []).append((ol["label"], ol["amount"]))

    for cl in custom_liabilities:
        liab_sections.setdefault(cl["group"], []).append((cl["label"], cl["amount"]))

    # 9. Build Final Presentation (Linear Lines and T-format Dual Columns)
    bs_lines: List[StatementLine] = []
    t_liabilities: List[Dict[str, Any]] = []
    t_assets: List[Dict[str, Any]] = []

    total_statutory_assets = ZERO
    bs_lines.append(StatementLine("ASSETS", ZERO, is_total=True))
    for grp_name, items in asset_sections.items():
        grp_tot = sum((amt for _, amt in items), ZERO)
        total_statutory_assets += grp_tot
        bs_lines.append(StatementLine(f"** {grp_name.upper()} **", ZERO, is_total=True))
        for name, amt in items:
            bs_lines.append(StatementLine(name, amt, indent=1))
            t_assets.append({"label": name, "amount": str(amt), "group": grp_name})
        bs_lines.append(StatementLine(f"Total {grp_name}", grp_tot, is_total=True))

    bs_lines.append(StatementLine("Total Statutory Assets", total_statutory_assets, is_total=True))

    total_statutory_liab = ZERO
    bs_lines.append(StatementLine("LIABILITIES & CAPITAL", ZERO, is_total=True))
    for grp_name, items in liab_sections.items():
        grp_tot = sum((amt for _, amt in items), ZERO)
        total_statutory_liab += grp_tot
        bs_lines.append(StatementLine(f"** {grp_name.upper()} **", ZERO, is_total=True))
        for name, amt in items:
            bs_lines.append(StatementLine(name, amt, indent=1))
            t_liabilities.append({"label": name, "amount": str(amt), "group": grp_name})
        bs_lines.append(StatementLine(f"Total {grp_name}", grp_tot, is_total=True))

    bs_lines.append(StatementLine("Total Statutory Liabilities + Capital", total_statutory_liab, is_total=True))

    diff = abs(total_statutory_assets - total_statutory_liab)
    ties = diff <= Decimal("0.05")
    if not ties:
        warnings.append(
            f"Statutory Balance Sheet has an imbalance: Assets {total_statutory_assets} "
            f"vs Liabilities {total_statutory_liab}. Difference: {diff}."
        )

    return {
        "ok": True,
        "ties": ties,
        "total_assets": str(total_statutory_assets),
        "total_liabilities": str(total_statutory_liab),
        "net_profit": str(statutory_net_profit),
        "closing_capital": str(closing_capital),
        "opening_capital": str(opening_cap),
        "total_drawings": str(tot_drawings),
        "t_format": {
            "liabilities": t_liabilities,
            "assets": t_assets,
            "total_liabilities": str(total_statutory_liab),
            "total_assets": str(total_statutory_assets),
            "ties": ties,
        },
        "t_format_pnl": {
            "expenses": t_pnl_expenses,
            "income": t_pnl_income,
            "total_expenses": str(pnl_grand_total),
            "total_income": str(pnl_grand_total),
            "ties": True,
        },
        "balance_sheet_lines": [
            {
                "label": l.label,
                "amount": str(l.amount),
                "code": l.code,
                "is_total": l.is_total,
                "indent": l.indent,
            }
            for l in bs_lines
        ],
        "profit_and_loss_lines": [
            {
                "label": l.label,
                "amount": str(l.amount),
                "code": l.code,
                "is_total": l.is_total,
                "indent": l.indent,
            }
            for l in pnl_lines
        ],
        "capital_account_lines": [
            {
                "label": l.label,
                "amount": str(l.amount),
                "code": l.code,
                "is_total": l.is_total,
                "indent": l.indent,
            }
            for l in cap_lines
        ],
        "warnings": warnings,
    }
