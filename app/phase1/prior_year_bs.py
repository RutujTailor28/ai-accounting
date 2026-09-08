"""Dynamic Prior Year Balance Sheet Parser.

Extracts line items, sections, and totals from any Indian Balance Sheet
(PDF horizontal T-format, PDF vertical format, Excel, or CSV).

Supports:
- Horizontal T-format (Liabilities on Left, Assets on Right)
- Vertical format (Schedule III or simple top-to-bottom list)
- Multi-page balance sheets
- Standard Indian head classifications (Fixed Assets, Deposits, Debtors,
  Cash & Bank, Stock, Capital, Loans, Creditors)
"""

from __future__ import annotations

import io
import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from app.phase1.contracts import ZERO, money


# Standard category keywords for classifying extracted balance sheet lines
_ASSET_CATEGORIES: Dict[str, Tuple[str, str]] = {
    # keyword: (category_name, default_ledger_code)
    "tool": ("Fixed Assets", "1110"),
    "equipment": ("Fixed Assets", "1110"),
    "furniture": ("Fixed Assets", "1110"),
    "fixture": ("Fixed Assets", "1110"),
    "vehic": ("Fixed Assets", "1110"),
    "car": ("Fixed Assets", "1110"),
    "bike": ("Fixed Assets", "1110"),
    "mobile": ("Fixed Assets", "1110"),
    "computer": ("Fixed Assets", "1110"),
    "laptop": ("Fixed Assets", "1110"),
    "plant": ("Fixed Assets", "1110"),
    "machinery": ("Fixed Assets", "1110"),
    "building": ("Fixed Assets", "1110"),
    "land": ("Fixed Assets", "1110"),
    "fixed asset": ("Fixed Assets", "1110"),
    "deposit": ("Deposits & Investments", "1230"),
    "investment": ("Deposits & Investments", "1230"),
    "shares": ("Deposits & Investments", "1230"),
    "mutual fund": ("Deposits & Investments", "1230"),
    "advance": ("Loans & Advances", "1220"),
    "loan & advance": ("Loans & Advances", "1220"),
    "debtor": ("Sundry Debtors", "1240"),
    "receivable": ("Sundry Debtors", "1240"),
    "stock": ("Closing Stock", "1060"),
    "inventory": ("Closing Stock", "1060"),
    "cash book": ("Cash in Hand", "1020"),
    "cash in hand": ("Cash in Hand", "1020"),
    "cash a/c": ("Cash in Hand", "1020"),
    "bank": ("Bank Account", "1010"),
    "tds receivable": ("Current Assets", "1210"),
    "gst input": ("Current Assets", "1210"),
}

_LIABILITY_CATEGORIES: Dict[str, Tuple[str, str]] = {
    "capital": ("Capital Account", "3010"),
    "partner": ("Capital Account", "3010"),
    "proprietor": ("Capital Account", "3010"),
    "reserves": ("Capital Account", "3020"),
    "surplus": ("Capital Account", "3020"),
    "creditor": ("Sundry Creditors", "2030"),
    "payable": ("Current Liabilities", "2020"),
    "loan": ("Loans & Borrowings", "2010"),
    "borrowing": ("Loans & Borrowings", "2010"),
    "overdraft": ("Loans & Borrowings", "2010"),
    "od a/c": ("Loans & Borrowings", "2010"),
    "gst payable": ("Duties & Taxes", "2110"),
    "tds payable": ("Duties & Taxes", "2120"),
    "duties": ("Duties & Taxes", "2130"),
    "taxes": ("Duties & Taxes", "2130"),
}


def classify_bs_head(label: str, section: str) -> Tuple[str, str]:
    """Determine group category and default ledger code for an extracted head."""
    label_lower = label.lower()
    if section == "asset":
        for kw, (grp, code) in _ASSET_CATEGORIES.items():
            if kw in label_lower:
                return grp, code
        return "Other Assets", "1250"
    else:
        for kw, (grp, code) in _LIABILITY_CATEGORIES.items():
            if kw in label_lower:
                return grp, code
        return "Other Liabilities", "2040"


def _parse_column_lines(
    text: str,
    section: str,
) -> List[Dict[str, Any]]:
    """Parse text lines from one column into structured balance sheet items."""
    items: List[Dict[str, Any]] = []
    current_group = "General"

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("===") or line.startswith("---"):
            continue
        if re.search(r"^(?:Account Name|Amount|\*+\s*TOTAL|\=+|\-+)", line, re.I):
            continue

        # Header detection like "** ASSETS :", "** DEPOSIT", "** ADVANCE", etc.
        grp_m = re.match(r"^\*{1,3}\s*([A-Za-z\s&/-]+?)(?:\s*:)?$", line)
        if grp_m:
            current_group = grp_m.group(1).strip().title()
            continue

        # Item line ending in an amount: "<Head Name> <Amount>"
        m = re.search(r"^(.*?)\s+([\d,]+\.\d{2}|[\d,]+)$", line)
        if m:
            name = m.group(1).strip()
            # Skip subtotal / total lines
            if re.search(r"^(?:Total|Sub\s*Total|Grand\s*Total)", name, re.I):
                continue
            amt_str = m.group(2).replace(",", "")
            amt = money(amt_str)
            if amt == ZERO:
                continue

            category, code = classify_bs_head(name, section)
            group_name = current_group if current_group != "General" else category

            items.append({
                "label": name,
                "amount": str(amt),
                "numeric_amount": amt,
                "group": group_name,
                "category": category,
                "code": code,
                "section": section,
            })

    return items


def parse_prior_balance_sheet(content: bytes, filename: str) -> Dict[str, Any]:
    """Dynamically parse a prior year Balance Sheet (PDF, XLSX, XLS, or CSV).

    Returns:
        Dict with:
          - ok: bool
          - client_name: str
          - as_at_date: str
          - liabilities: List[Dict]
          - assets: List[Dict]
          - total_liabilities: str
          - total_assets: str
          - opening_capital: str
          - ties: bool
          - warnings: List[str]
    """
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return _parse_pdf_balance_sheet(content, filename)
    elif lower_name.endswith((".xlsx", ".xls", ".csv")):
        return _parse_tabular_balance_sheet(content, filename)
    else:
        return {
            "ok": False,
            "errors": [f"Unsupported file format: {filename}. Please upload PDF or Excel."],
        }


def _parse_pdf_balance_sheet(content: bytes, filename: str) -> Dict[str, Any]:
    """Parse Indian Balance Sheet PDF dynamically."""
    import pdfplumber

    liabilities: List[Dict[str, Any]] = []
    assets: List[Dict[str, Any]] = []
    client_name = ""
    as_at_date = ""
    warnings: List[str] = []

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                full_text = page.extract_text() or ""

                # Extract Client Name (first non-empty line of page 1 if not header)
                if not client_name:
                    first_lines = [l.strip() for l in full_text.split("\n") if l.strip()]
                    if first_lines:
                        client_name = re.sub(r"\s+\d{4}$", "", first_lines[0]).strip()

                # Extract As At date (e.g. "AS AT 31-03-2024" or "AS ON 31/03/2024")
                if not as_at_date:
                    date_m = re.search(r"\bAS\s+(?:AT|ON)\s+([0-9]{1,2}[-/.][0-9]{1,2}[-/.][0-9]{2,4})", full_text, re.I)
                    if date_m:
                        as_at_date = date_m.group(1).strip()

                words = page.extract_words()
                if not words:
                    continue

                # Standard Indian Balance Sheet: Left half is Liabilities, Right half is Assets
                header_y: Optional[float] = None
                mid_x: float = page.width / 2.0
                footer_y: float = page.height - 30.0

                by_row: Dict[int, List[Dict[str, Any]]] = {}
                for w in words:
                    r_top = int(round(w["top"] / 4.0) * 4)
                    by_row.setdefault(r_top, []).append(w)

                for r_top, row_words in sorted(by_row.items()):
                    row_text = " ".join(w["text"] for w in sorted(row_words, key=lambda x: x["x0"]))
                    compact = re.sub(r"\s+", "", row_text).upper()
                    if ("LIABILIT" in compact or compact.startswith("LIAB")) and "ASSET" in compact:
                        # Found column header row (e.g. "LIABILITIES ASSETS")
                        header_y = row_words[0]["top"] + 16.0
                        break
                    if "ACCOUNT" in compact and "AMOUNT" in compact:
                        header_y = row_words[0]["top"] + 14.0
                        break

                if header_y is None:
                    header_y = 60.0

                # Find bottom Grand Total row (scan from bottom upwards)
                for r_top, row_words in sorted(by_row.items(), reverse=True):
                    if r_top > header_y:
                        row_text = " ".join(w["text"] for w in row_words)
                        if re.search(r"(?:\*+\s*TOTAL|GRAND\s*TOTAL|\bTOTAL\b)", row_text, re.I) and any(
                            re.search(r"\d+\.\d{2}", w["text"]) for w in row_words
                        ):
                            footer_y = row_words[0]["top"] - 2.0
                            break

                left_crop = page.crop((0, header_y, mid_x, footer_y))
                right_crop = page.crop((mid_x, header_y, page.width, footer_y))

                left_items = _parse_column_lines(left_crop.extract_text() or "", "liability")
                right_items = _parse_column_lines(right_crop.extract_text() or "", "asset")

                liabilities.extend(left_items)
                assets.extend(right_items)

    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"Could not parse Balance Sheet PDF: {exc}"],
        }

    total_liab = money(sum((Decimal(l["amount"]) for l in liabilities), ZERO))
    total_asset = money(sum((Decimal(a["amount"]) for a in assets), ZERO))

    opening_cap = ZERO
    for l in liabilities:
        if (
            l["category"] == "Capital Account"
            or "capital" in l["label"].lower()
            or "capital" in l.get("group", "").lower()
        ):
            l["category"] = "Capital Account"
            l["code"] = "3010"
            opening_cap += Decimal(l["amount"])

    if opening_cap == ZERO:
        other_liab = sum(
            (Decimal(l["amount"]) for l in liabilities if l["category"] != "Capital Account"), ZERO
        )
        opening_cap = money(total_asset - other_liab)

    ties = abs(total_liab - total_asset) <= Decimal("1.00")
    if not ties:
        warnings.append(
            f"Extracted Liabilities ({total_liab}) and Assets ({total_asset}) differ. "
            f"Please verify or adjust the items in the review table."
        )

    return {
        "ok": True,
        "filename": filename,
        "client_name": client_name,
        "as_at_date": as_at_date,
        "liabilities": [
            {k: v for k, v in item.items() if k != "numeric_amount"}
            for item in liabilities
        ],
        "assets": [
            {k: v for k, v in item.items() if k != "numeric_amount"}
            for item in assets
        ],
        "total_liabilities": str(total_liab),
        "total_assets": str(total_asset),
        "opening_capital": str(opening_cap),
        "ties": ties,
        "warnings": warnings,
    }


def _parse_tabular_balance_sheet(content: bytes, filename: str) -> Dict[str, Any]:
    """Parse Balance Sheet from Excel or CSV."""
    import pandas as pd

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content), header=None, dtype=str)
        else:
            df = pd.read_excel(io.BytesIO(content), header=None, dtype=str)
    except Exception as exc:
        return {"ok": False, "errors": [f"Failed to read file: {exc}"]}

    df = df.fillna("")
    liabilities: List[Dict[str, Any]] = []
    assets: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row if str(v).strip()]
        if len(vals) >= 4:
            name_l, amt_l, name_r, amt_r = vals[0], vals[1], vals[2], vals[3]
            m_l = re.search(r"([\d,]+\.?\d*)", amt_l)
            m_r = re.search(r"([\d,]+\.?\d*)", amt_r)
            if m_l and not name_l.lower().startswith("total"):
                amt = money(m_l.group(1).replace(",", ""))
                cat, code = classify_bs_head(name_l, "liability")
                liabilities.append({"label": name_l, "amount": str(amt), "group": cat, "category": cat, "code": code, "section": "liability"})
            if m_r and not name_r.lower().startswith("total"):
                amt = money(m_r.group(1).replace(",", ""))
                cat, code = classify_bs_head(name_r, "asset")
                assets.append({"label": name_r, "amount": str(amt), "group": cat, "category": cat, "code": code, "section": "asset"})

    total_liab = money(sum((Decimal(l["amount"]) for l in liabilities), ZERO))
    total_asset = money(sum((Decimal(a["amount"]) for a in assets), ZERO))

    return {
        "ok": True,
        "filename": filename,
        "client_name": "",
        "as_at_date": "",
        "liabilities": liabilities,
        "assets": assets,
        "total_liabilities": str(total_liab),
        "total_assets": str(total_asset),
        "opening_capital": str(total_liab),
        "ties": total_liab == total_asset,
        "warnings": [],
    }
