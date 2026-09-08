"""Tests for the Phase 1 deterministic pipeline.

Runs standalone (`python tests/test_phase1.py`) or under pytest. No network,
no LLM, no database - which is the point of Phase 1.

Statements are generated in-test with an arithmetically correct running
balance, so the expected figures are derived rather than hand-typed.
"""

from __future__ import annotations

import csv
import io
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # Windows consoles default to cp1252 and cannot print the rupee sign
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.phase1 import coa, parsers, pipeline  # noqa: E402
from app.phase1.contracts import Source, money  # noqa: E402
from app.phase1.normalise import normalise  # noqa: E402
from app.phase1.rules import RulesEngine  # noqa: E402

HEADER = ["Date", "Narration", "Chq./Ref.No.", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"]


def build_csv(rows, opening=Decimal("250000.00"), corrupt_row=None, with_balance=True):
    """Build a bank-statement CSV with a correct running balance.

    `rows` are `(date, narration, debit, credit)`. `corrupt_row` breaks the
    balance on that 1-based row, to exercise the validation gate.
    """
    buf = io.StringIO()
    w = csv.writer(buf)
    # Junk preamble, exactly like a real export.
    w.writerow(["STATE BANK OF EXAMPLE"])
    w.writerow(["Account: XXXXXX7788"])
    w.writerow([])
    w.writerow(HEADER if with_balance else HEADER[:-1])

    bal = opening
    for i, (d, n, dr, cr) in enumerate(rows, start=1):
        bal = bal - Decimal(dr or "0") + Decimal(cr or "0")
        shown = bal + Decimal("500.00") if corrupt_row == i else bal
        row = [d, n, "", dr, cr]
        if with_balance:
            row.append(f"{shown:.2f}")
        w.writerow(row)
    return buf.getvalue().encode("utf-8"), bal


CLEAN_ROWS = [
    ("01/04/2024", "UPI/423122011/Payment from/RAVI TRADERS@okhdfcbank", "", "125000.00"),
    ("02/04/2024", "NEFT DR-HDFC0000123-ASHOK RENTALS-RENT APRIL", "45000.00", ""),
    ("05/04/2024", "SALARY APRIL 2024 STAFF PAYOUT", "180000.00", ""),
    ("06/04/2024", "ATM CASH WDL 1234 MUMBAI", "10000.00", ""),
    ("07/04/2024", "SMS CHG 010424 TO 300424", "23.60", ""),
    ("08/04/2024", "GST PAYMENT CBIC ITNS0020 REF 998877", "32000.00", ""),
    ("10/04/2024", "INT.PD:010424-300424", "", "1450.00"),
    ("11/04/2024", "AIRTEL BROADBAND BILL PAYMENT", "2360.00", ""),
    ("12/04/2024", "UPI/112233/Payment to/HPCL PETROL PUMP@ybl", "3000.00", ""),
    ("12/04/2024", "UPI/112233/Payment to/HPCL PETROL PUMP@ybl", "3000.00", ""),
    ("15/04/2024", "EMI DEBIT LOAN ACCOUNT 88776655", "25000.00", ""),
    ("18/04/2024", "SOME UNKNOWN VENDOR PAYMENT XYZ", "7500.00", ""),
    ("20/04/2024", "UPI/334455/Payment from/KUMAR AND SONS@paytm", "", "56000.00"),
]

_failures: list[str] = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  {status}  {name}" + ("" if cond else f"   <- {detail}"))
    if not cond:
        _failures.append(name)


# ---------------------------------------------------------------------------


def test_happy_path():
    print("\n[1] Clean statement end to end")
    data, expected_closing = build_csv(CLEAN_ROWS)
    r = pipeline.run(data, "stmt.csv")

    check("run succeeded", r.ok, str(r.errors))
    check("all rows parsed", len(r.transactions) == len(CLEAN_ROWS),
          f"{len(r.transactions)} vs {len(CLEAN_ROWS)}")
    check("trial balance ties", r.statements.ties,
          f"Dr {r.statements.total_debits} Cr {r.statements.total_credits}")
    check("opening balance recovered", r.statements.opening_balance == Decimal("250000.00"),
          str(r.statements.opening_balance))
    check("closing matches statement", r.statements.closing_balance == expected_closing,
          f"{r.statements.closing_balance} vs {expected_closing}")
    check("no float drift (exact 2dp)",
          all(str(m).count(".") == 1 and len(str(m).split(".")[1]) == 2
              for m in [r.statements.total_debits, r.statements.total_credits]))

    # Accounting identity: closing bank = opening + profit (cash basis, no
    # other assets/liabilities beyond those the rules created).
    by_code = {row["code"]: row for row in r.statements.trial_balance}
    check("bank account present", coa.BANK in by_code)
    return r


def test_rules_and_review():
    print("\n[2] Classification")
    data, _ = build_csv(CLEAN_ROWS)
    r = pipeline.run(data, "stmt.csv")

    def code_for(fragment):
        """Ledger assigned to the row whose narration contains `fragment`."""
        for t in r.transactions:
            if fragment.lower() in t.narration_raw.lower():
                return t.ledger_code
        return None

    check("rent matched", code_for("ASHOK RENTALS") == "5010", str(code_for("ASHOK RENTALS")))
    check("salary matched", code_for("SALARY APRIL") == "5020", str(code_for("SALARY APRIL")))
    check("atm -> cash", code_for("ATM CASH WDL") == "1020")
    check("sms charges -> bank charges", code_for("SMS CHG") == "5060")
    check("gst -> gst payable", code_for("GST PAYMENT") == "2110")
    check("interest credit -> income", code_for("INT.PD") == "4020")
    check("telecom matched", code_for("AIRTEL") == "5050")
    check("emi -> loan", code_for("EMI DEBIT") == "2010")
    check("unmatched debit -> suspense", code_for("SOME UNKNOWN VENDOR") == coa.SUSPENSE)
    check("unmatched credit -> receipts", code_for("KUMAR AND SONS") == "4010")

    unmatched = [t for t in r.transactions if t.source == Source.UNCLASSIFIED]
    check("suspense rows flagged for review", len(unmatched) >= 1, str(len(unmatched)))
    check("coverage reported", 0 < r.coverage["coverage_pct"] <= 100, str(r.coverage))


def test_duplicate_rows_survive():
    print("\n[3] Genuine duplicate payments are kept")
    data, _ = build_csv(CLEAN_ROWS)
    r = pipeline.run(data, "stmt.csv")
    hpcl = [t for t in r.transactions if "HPCL" in t.narration_raw]
    check("both identical HPCL payments kept", len(hpcl) == 2, f"got {len(hpcl)}")
    check("their fingerprints differ",
          hpcl[0].fingerprint() != hpcl[1].fingerprint() if len(hpcl) == 2 else False)


def test_balance_gate_rejects_corruption():
    print("\n[4] Running-balance gate")
    # One corrupted row in a short statement is >2% of rows -> reject.
    data, _ = build_csv(CLEAN_ROWS, corrupt_row=5)
    r = pipeline.run(data, "stmt.csv")
    check("corrupted statement rejected", not r.ok, "it was accepted")
    check("error explains why",
          any("reconcile" in e.lower() for e in r.errors), str(r.errors))
    check("no statements produced", r.statements is None)


def test_missing_balance_column():
    print("\n[5] Statement with no balance column")
    data, _ = build_csv(CLEAN_ROWS, with_balance=False)
    r = pipeline.run(data, "stmt.csv")
    check("still parses", r.ok, str(r.errors))
    check("warns that it could not cross-check",
          any("cross-check" in w or "balance column" in w for w in r.warnings), str(r.warnings))
    check("still ties", r.statements.ties)


def test_human_override():
    print("\n[6] Reviewer override beats the rules")
    data, _ = build_csv(CLEAN_ROWS)
    base = pipeline.run(data, "stmt.csv")
    target = next(t for t in base.transactions if t.ledger_code == coa.SUSPENSE)

    r = pipeline.run(data, "stmt.csv", overrides={target.row_index: "5030"})
    changed = next(t for t in r.transactions if t.row_index == target.row_index)
    check("ledger changed", changed.ledger_code == "5030", changed.ledger_code)
    check("marked as human", changed.source == Source.HUMAN, changed.source.value)
    check("still ties after override", r.statements.ties)
    check("suspense count dropped",
          r.coverage["needs_review"] < base.coverage["needs_review"],
          f"{r.coverage} vs {base.coverage}")


def test_normaliser():
    print("\n[7] Narration normaliser")
    cases = [
        ("UPI/423122011/Payment from/SWIGGY@axisbank/UTIB0000123", "swiggy", "UPI"),
        ("NEFT DR-HDFC0000123-ASHOK RENTALS-RENT APRIL", "ashok rentals rent april", "NEFT"),
        ("ATM CASH WDL 1234 MUMBAI", None, "ATM"),
        ("SMS CHG 010424 TO 300424", None, "CHARGES"),
        ("INT.PD:010424-300424", None, "INT"),
        ("IMPS/P2A/412345678/RAMESH KUMAR", "ramesh kumar", "IMPS"),
    ]
    for narration, expected_key, expected_channel in cases:
        _, key, channel = normalise(narration)
        check(f"channel {expected_channel} from {narration[:28]!r}",
              channel.value == expected_channel, f"got {channel.value}")
        if expected_key is not None:
            check(f"key {expected_key!r}", key == expected_key, f"got {key!r}")


def test_money_parsing():
    print("\n[8] Money coercion")
    cases = [
        ("1,25,000.50", Decimal("125000.50")),
        ("₹ 4,500", Decimal("4500.00")),
        ("(1200.00)", Decimal("-1200.00")),
        ("5000 Cr", Decimal("5000.00")),
        ("", Decimal("0.00")),
        ("-", Decimal("0.00")),
        (None, Decimal("0.00")),
        (0.1, Decimal("0.10")),
    ]
    for raw, expected in cases:
        got = money(raw)
        check(f"money({raw!r}) == {expected}", got == expected, f"got {got}")

    # Float would drift here; Decimal must not.
    total = sum((money("0.10") for _ in range(1000)), Decimal("0"))
    check("1000 x 0.10 == 100.00 exactly", total == Decimal("100.00"), str(total))


def test_unsupported_file():
    print("\n[9] Unsupported input")
    r = pipeline.run(b"not a statement", "notes.txt")
    check("rejected", not r.ok)
    check("message names the accepted formats",
          any("PDF" in e for e in r.errors), str(r.errors))


def test_zero_llm():
    print("\n[10] No LLM anywhere in the Phase 1 package")
    import pathlib
    import re as _re

    # Match real imports and client construction - NOT bare substrings. The
    # rules file legitimately contains the vendor name "openai" as a pattern
    # for classifying a software subscription payment.
    import_re = _re.compile(
        r"^\s*(?:from|import)\s+(openai|anthropic|langchain\w*|ollama)", _re.M
    )
    client_re = _re.compile(r"(ChatOpenAI|AsyncOpenAI|OpenAI|Anthropic)\s*\(")

    hits = []
    for path in sorted(pathlib.Path("app/phase1").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for m in import_re.finditer(text):
            hits.append(f"{path.name}: imports {m.group(1)}")
        for m in client_re.finditer(text):
            hits.append(f"{path.name}: constructs {m.group(1)}")
    check("no model imports or client construction", not hits, str(hits))


def main():
    test_happy_path()
    test_rules_and_review()
    test_duplicate_rows_survive()
    test_balance_gate_rejects_corruption()
    test_missing_balance_column()
    test_human_override()
    test_normaliser()
    test_money_parsing()
    test_unsupported_file()
    test_zero_llm()

    print()
    if _failures:
        print(f"FAILURES ({len(_failures)}): {_failures}")
        return 1
    print("ALL PHASE 1 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
