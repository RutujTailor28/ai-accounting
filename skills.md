---
name: bank-statement-accountant
description: Turns a business's bank statement(s) — PDF, scanned/photographed, Excel, or CSV — into a Trading & Profit and Loss Account, Balance Sheet, and Capital Account. Use this whenever the user shares a bank statement (or several, for the same business) and wants books prepared, wants to know their profit/loss, wants a balance sheet, wants a capital account reconciled, or asks for "accounting" or "bookkeeping" from raw bank data. Also trigger when the user is an accountant/CA or works at an accounting/bookkeeping firm and mentions preparing financials for a client from bank statements. Do NOT trigger for pure data-entry tasks (just reformat this CSV) with no accounting statement requested, or for personal (non-business) budget tracking.
---

# Bank Statement → Financial Statements

Turns raw bank statement transactions into draft accounting statements: **Trading & Profit and Loss Account**, **Balance Sheet**, and **Capital Account** (or Partners'/Shareholders' equivalent). The output is a *draft for review* — bank-derived books are never a substitute for full double-entry accounting, and the final deliverable must say so plainly. Treat this as work a junior accountant hands to a senior for review, not as certified financials.

## Why this is harder than "sum some columns"

A bank statement only shows cash that moved through one account. It does not know:
- what's owed to the business (debtors) or by it (creditors) — accrual items invisible to a bank feed
- stock/inventory on hand
- fixed assets bought in cash, via a different account, or before the statement period
- which withdrawals are business expense vs. personal drawings, unless the narration says so or the user does
- opening balances of capital, loans, or the bank account itself, unless supplied

Getting these wrong doesn't just look sloppy — it silently misstates profit and the capital account, and if the user hands this to a client or files taxes off it, that's a real-world consequence. So the operating principle throughout is: **classify confidently where the narration is genuinely clear, and flag everything else for the user rather than guessing.** A statement with 15 flagged items the user resolves in two minutes beats a fully-populated statement built on invented assumptions.

## Workflow

### 1. Establish the basics before extracting anything

Ask (briefly, one round) whatever isn't already clear from context:
- **Entity type**: sole proprietorship, partnership, or company. This changes the equity section entirely — see `references/entity_formats.md`. If genuinely unknown, default to sole proprietorship (simplest) and say so.
- **Period**: statement date range, or the financial year this covers.
- **How many statements / accounts**, and whether they belong to the same business (needed for consolidation and transfer detection, step 3).
- **Opening balances**, if the user has them: opening capital, opening bank balance, any outstanding loans. If not available, say the statements will be treated as if the business started with the bank's opening balance as opening capital — flag this assumption in the output, don't bury it.

Don't turn this into a long form — infer what's inferable (e.g., a single PDF titled "XYZ Enterprises Current A/c" implies sole proprietorship unless told otherwise) and only ask what actually changes the output.

### 2. Extract transactions

Route by input type — check `/mnt/skills/public/pdf-reading/SKILL.md` or `/mnt/skills/public/xlsx/SKILL.md` first if you need a refresher on extraction mechanics for that format:

- **Excel/CSV**: read directly with pandas.
- **Text-based PDF**: extract tables (`pdftotext -layout` or a table-extraction pass); bank statement tables are usually regular (Date, Narration, Ref, Debit, Credit, Balance columns) but column order and headers vary by bank.
- **Scanned/photographed statements**: rasterize pages and read them visually (per `pdf-reading` skill guidance) — do not attempt OCR-free text extraction on an image-only PDF, you'll get garbage or nothing.

Normalize every transaction to: `date, narration, debit, credit, balance, source_account`. Keep the original narration verbatim — you'll need its exact wording for classification and the user will want to spot-check it.

**Reconcile before trusting the extraction.** For each account: opening balance + Σcredits − Σdebits should equal the closing balance. If it doesn't, you dropped, duplicated, or misread rows — go find the discrepancy before moving on. This one check catches the majority of extraction errors and takes seconds; skipping it means every downstream number is suspect.

### 3. Handle multiple accounts

If there's more than one statement for the same business, match transfers between the business's own accounts (same-day or near-same-day, matching amount, opposite direction) and tag them `internal_transfer`. These are not income or expense — including them would double-count. Keep them visible in the working file but exclude from P&L and from revenue/expense totals.

If the statements are for genuinely different entities, don't consolidate — produce separate statements and say so.

### 4. Classify every transaction

Use `references/classification_rules.md` for the keyword/pattern heuristics and the full ledger head list. In short, every transaction lands in one of:
**Revenue** · **Purchases/COGS** · **Direct expense** (freight, direct wages) · **Indirect/operating expense** (rent, salaries, utilities, professional fees, bank charges, interest paid) · **Capital introduced** · **Drawings** · **Loan proceeds** · **Loan repayment (principal)** · **Interest paid/received on loan** · **Fixed asset purchase** · **Statutory payment** (GST/TDS/advance tax) · **Internal transfer** · **Unclassified**

For each transaction, decide from the narration text and amount pattern. A few examples: `UPI-SWIGGY` next to otherwise business-only activity is ambiguous — food could be a business expense (client meeting) or personal, so it's a drawings-vs-expense judgment call, flag it. `NEFT SALARY XYZ` is an indirect expense, confidently. A large round-number inward transfer from what looks like a personal account name, early in the period, with nothing to match it to, is very plausibly capital introduced.

Anything you can't classify with real confidence goes to **Unclassified**, with your best guess noted, not silently dropped into "Other Expenses." Build a running list of every flagged/unclassified item — you'll present this to the user before finalizing (step 6), because these judgment calls are exactly the ones that most affect profit and capital.

### 5. Build the statements

Work in this order, each feeding the next:

1. **Trading Account** (if the business has purchases/COGS): Sales − Purchases − Direct expenses = Gross Profit.
2. **Profit & Loss Account**: Gross Profit + other income − indirect expenses = Net Profit.
3. **Capital Account**: Opening Capital + Capital Introduced + Net Profit − Drawings = Closing Capital. For a partnership, split by each partner's identified contributions/drawings, or by the profit-sharing ratio if the user gives one — see `references/entity_formats.md`. For a company, there's no capital account in this sense; direct to share capital and reserves & surplus instead.
4. **Balance Sheet**: Bank balance (closing, from statement) as an asset; Closing Capital and any identified loan balances as liabilities/equity. Explicitly list what's *missing* (debtors, creditors, stock, non-bank fixed assets) as a visible line or note — a balance sheet that silently balances only because unknown items are absent is misleading, not clean.

Check the arithmetic actually ties out: Balance Sheet assets = liabilities + equity, and the capital account closing balance matches what's shown as equity. If it doesn't tie, find why before presenting anything.

### 6. Surface flagged items before finalizing

Before generating final output, show the user the list of Unclassified/flagged transactions in-chat (narration, amount, your best guess) and ask them to confirm or correct. This is the single highest-leverage step for accuracy — skipping it to save a round trip means shipping a P&L built partly on guesses. If there are more than ~15-20 flagged items, group by pattern (e.g., "6 UPI payments to individuals, ₹2,000-8,000, no clear business purpose") rather than listing every row.

### 7. Produce output

Read `/mnt/skills/public/xlsx/SKILL.md` and `/mnt/skills/public/docx/SKILL.md` before generating files — this skill defines the accounting logic, those skills own the actual file mechanics (formulas, formatting, recalculation).

**Excel workbook** — sheets, in this order:
- `Transactions` — every transaction, normalized, with assigned ledger head and source account. This is the audit trail; nothing should be traceable-back-to-here-and-missing.
- `Classification Notes` — every flagged/unclassified item and how it was resolved.
- `Trading & P&L Account` — formulas referencing `Transactions` (e.g. `SUMIFS`), not hardcoded totals, so the sheet recalculates if a classification changes.
- `Balance Sheet` — same principle; formulas back to the P&L and Capital Account sheets, plus a clearly labeled note section for the missing items from step 5.
- `Capital Account` — opening, movements, closing, formula-linked to P&L net profit.

Follow the xlsx skill's financial-model conventions (blue = hardcoded inputs, black = formulas, yellow fill = assumptions/flagged items) and run `recalc.py` before delivering — a workbook with unresolved formula cells is worse than one with none.

**Word/PDF report** — a formatted, presentation-ready version of the three statements (standard vertical format per `references/entity_formats.md`), with a short front-page or footer note on scope and limitations (draft, bank-derived, unaudited, entity type assumed, opening balances assumed if applicable).

Both outputs must open with (or otherwise clearly carry) the same limitations note — don't let the polished formatting imply more certainty than the source data supports.

## Reference files

- `references/classification_rules.md` — keyword patterns and worked examples for every ledger head; read this before classifying transactions.
- `references/entity_formats.md` — statement formats and equity-section differences for sole proprietorship, partnership, and company; read this when building the Capital Account / equity section.
- `references/limitations.md` — the standard limitations note, and what additional inputs from the user would upgrade this from a bank-derived draft toward real financials (useful to offer once, not to nag about repeatedly).