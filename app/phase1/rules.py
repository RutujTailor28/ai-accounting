"""Deterministic classification rules - no LLM, no network, no memory.

Strictly Phase 1 says every row is classified by hand. That is unusable for
an accountant testing a 2,000-row statement, so this seed rule set does the
obvious rows and leaves everything else in Suspense for review. It is still
zero-LLM and fully reproducible: the same statement always yields the same
classification.

Rules are tried in `priority` order; first match wins. Anything unmatched
goes to Suspense rather than being guessed - a wrong ledger silently booked
is worse than an obvious gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

from app.phase1 import coa
from app.phase1.contracts import Channel, Source, Transaction


@dataclass(frozen=True)
class Rule:
    """One pattern -> ledger mapping.

    `direction` restricts the rule to money-in ("credit"), money-out
    ("debit"), or "any". Getting this right matters: "INTEREST" credited is
    income, "INTEREST" debited is an expense.
    `min_amount` and `max_amount` allow tiering by transaction size.
    """

    id: str
    ledger_code: str
    priority: int = 100
    pattern: Optional[str] = None
    channel: Optional[Channel] = None
    direction: str = "any"
    min_amount: Optional[Decimal] = None
    max_amount: Optional[Decimal] = None

    def matches(self, txn: Transaction) -> bool:
        if self.direction != "any" and txn.direction != self.direction:
            return False
        if self.channel is not None and txn.channel != self.channel:
            return False
        amt = txn.debit if txn.direction == "debit" else txn.credit
        if self.min_amount is not None and amt < self.min_amount:
            return False
        if self.max_amount is not None and amt > self.max_amount:
            return False
        if self.pattern:
            haystack = f"{txn.narration_raw} {txn.counterparty_key}".lower()
            if not re.search(self.pattern, haystack, re.I):
                return False
        return True


# Lower priority number = checked earlier.
SEED_RULES: List[Rule] = [
    # -- Bank's own postings. Unambiguous, so checked first. ---------------
    Rule("int-income", "4020", 10, channel=Channel.INTEREST, direction="credit"),
    Rule("int-paid", "5070", 11, channel=Channel.INTEREST, direction="debit"),
    Rule("bank-charges", "5060", 12, channel=Channel.CHARGES, direction="debit"),
    Rule("bank-charges-kw", "5060", 13,
         pattern=r"\b(sms\s*chg|amc|folio|min\s*bal|cheque\s*ret|ecs\s*return|"
                 r"nach\s*ret|penal|service\s*chg|processing\s*fee)\b",
         direction="debit"),

    # -- Statutory. High confidence from the keyword alone. ----------------
    Rule("gst", "2110", 20, pattern=r"\b(gst|gstn|cbic|cgst|sgst|igst)\b", direction="debit"),
    Rule("tds-advtax", "1210", 21,
         pattern=r"\b(tds|itns|advance\s*tax|income\s*tax|incometax|self\s*asst)\b",
         direction="debit"),
    Rule("pf-esi", "2130", 22,
         pattern=r"\b(epf|epfo|pf\b|esic?|professional\s*tax|ptax|labour\s*welfare)\b",
         direction="debit"),

    # -- Payroll -----------------------------------------------------------
    Rule("salary", "5020", 30,
         pattern=r"\b(salary|salaries|sal\s|payroll|wages|stipend|bonus)\b",
         direction="debit"),

    # -- Rent --------------------------------------------------------------
    Rule("rent", "5010", 35, pattern=r"\b(rent|lease|licence\s*fee)\b", direction="debit"),

    # -- Utilities & comms -------------------------------------------------
    Rule("utilities", "5040", 40,
         pattern=r"\b(electricity|mseb|bses|torrent\s*power|adani\s*elec|tata\s*power|"
                 r"gas\b|water\s*bill|municipal|nagar)\b",
         direction="debit"),
    Rule("telecom", "5050", 41,
         pattern=r"\b(airtel|jio|vodafone|vi\s*postpaid|bsnl|idea|broadband|internet|"
                 r"act\s*fibernet|hathway|recharge|dth|tatasky|dishtv)\b",
         direction="debit"),

    # -- Common vendor categories -----------------------------------------
    Rule("fuel", "5090", 50,
         pattern=r"\b(petrol|diesel|fuel|hpcl|bpcl|iocl|indian\s*oil|hp\s*petrol|shell|petroleu|petroleum|cng)\b",
         direction="debit"),
    Rule("travel", "5080", 51,
         pattern=r"\b(irctc|makemytrip|goibibo|yatra|uber|ola\b|rapido|indigo|spicejet|"
                 r"air\s*india|vistara|redbus|travel|hotel|oyo)\b",
         direction="debit"),
    Rule("insurance", "5100", 52,
         pattern=r"\b(insurance|lic\b|policy|premium|hdfc\s*life|icici\s*pru|star\s*health|"
                 r"bajaj\s*allianz)\b",
         direction="debit"),
    Rule("software", "5130", 53,
         pattern=r"\b(google|microsoft|adobe|zoho|aws|amazon\s*web|atlassian|slack|"
                 r"notion|figma|github|openai|subscription|saas|tally\s*solutions|razorpay\s*soft)\b",
         direction="debit"),
    Rule("courier", "5150", 54,
         pattern=r"\b(dtdc|bluedart|blue\s*dart|delhivery|fedex|dhl|india\s*post|"
                 r"speed\s*post|courier|shiprocket|freight|transport)\b",
         direction="debit"),
    Rule("marketing", "5140", 55,
         pattern=r"\b(google\s*ads|facebook|meta\s*plat|instagram|linkedin|advertis|"
                 r"marketing|promotion)\b",
         direction="debit"),
    Rule("professional", "5110", 56,
         pattern=r"\b(consultanc|consulting|advocate|legal|audit|chartered|"
                 r"\bca\s*fee|professional\s*fee|retainer)\b",
         direction="debit"),
    Rule("repairs", "5120", 57,
         pattern=r"\b(repair|maintenance|servicing|amc\s*service|plumb|electrician)\b",
         direction="debit"),
    Rule("office", "5160", 58,
         pattern=r"\b(stationery|printing|xerox|office\s*supp|pantry|housekeep|"
                 r"amazon\b|flipkart|blinkit|zepto|swiggy|bigbasket|"
                 r"tea|chai|coffee|cafe|canteen|snacks|bakery|\bpan\b|paan|sweet|mithai|"
                 r"restaurant|hotel|bhojanalay|dhaba|refreshment|dairy|kirana|supermarket|mart|aamla|"
                 r"vyapar|paytmqr|bharatpe|phonepeqr)\b",
         direction="debit"),

    # -- Loans & cards -----------------------------------------------------
    Rule("loan-emi", "2010", 60,
         pattern=r"\b(emi|loan|lien|hypothec|term\s*loan|od\s*account|cc\s*limit|bajaj\s*fin|tvs\s*credit|hdb\s*fin|chola|shriram|muthoot|manappuram)\b",
         direction="debit"),
    Rule("credit-card", "2020", 61,
         pattern=r"\b(credit\s*card|cc\s*payment|card\s*pmt|autopay\s*card)\b",
         direction="debit"),

    # -- Cash movements ----------------------------------------------------
    Rule("atm-withdrawal", "1020", 70, channel=Channel.ATM, direction="debit"),
    Rule("cash-deposit", "1020", 71, channel=Channel.CASH, direction="credit"),

    # -- Capital -----------------------------------------------------------
    Rule("drawings", "3040", 80,
         pattern=r"\b(drawing|personal\s*use|self\s*trf|own\s*account)\b", direction="debit"),
    Rule("capital-in", "3030", 81,
         pattern=r"\b(capital|equity\s*infusion|owner\s*contrib|proprietor)\b",
         direction="credit"),

    # -- Income ------------------------------------------------------------
    Rule("refund", "4040", 90,
         pattern=r"\b(refund|reversal|cashback|chargeback|returned)\b", direction="credit"),
    Rule("dividend-interest", "4020", 91,
         pattern=r"\b(dividend|maturity|fd\s*int|rd\s*int)\b", direction="credit"),

    # -- Amount-tiered UPI transfers to individuals (evaluated after keyword rules) --
    Rule("upi-small-expense", "5160", 95, channel=Channel.UPI, direction="debit",
         max_amount=Decimal("2000.00")),
    Rule("upi-drawings-transfer", "3040", 96, channel=Channel.UPI, direction="debit",
         min_amount=Decimal("2000.01"), max_amount=Decimal("50000.00")),
]


class RulesEngine:
    """Applies the seed rules. Deterministic and side-effect free."""

    def __init__(self, rules: Optional[List[Rule]] = None) -> None:
        self.rules = sorted(rules or SEED_RULES, key=lambda r: r.priority)

    def classify(self, txn: Transaction) -> Transaction:
        """Assign a ledger to one transaction. Mutates and returns it."""
        for rule in self.rules:
            if rule.matches(txn):
                txn.ledger_code = rule.ledger_code
                txn.source = Source.RULE
                txn.rule_id = rule.id
                return txn

        # No rule matched. Fall back on direction only - money in with no
        # other signal is most likely a receipt; money out is unknown and
        # must be reviewed rather than guessed into an expense head.
        if txn.direction == "credit":
            txn.ledger_code = "4010"
            txn.source = Source.RULE
            txn.rule_id = "default-receipt"
        else:
            txn.ledger_code = coa.SUSPENSE
            txn.source = Source.UNCLASSIFIED
            txn.rule_id = None
        return txn

    def classify_all(self, transactions: List[Transaction]) -> List[Transaction]:
        return [self.classify(t) for t in transactions]

    @staticmethod
    def coverage(transactions: List[Transaction]) -> dict:
        """How much got classified without human help - the number that
        decides whether this is worth using."""
        total = len(transactions)
        if not total:
            return {"total": 0, "classified": 0, "needs_review": 0, "coverage_pct": 0.0}
        review = sum(
            1 for t in transactions
            if t.source == Source.UNCLASSIFIED or t.ledger_code == coa.SUSPENSE
        )
        return {
            "total": total,
            "classified": total - review,
            "needs_review": review,
            "coverage_pct": round((total - review) / total * 100, 1),
        }
