# Transaction Classification Rules

Every transaction gets one ledger head. These are heuristics to reason with, not a lookup table to apply mechanically — narration formats vary a lot by bank, and the same keyword can mean different things depending on context (who the business is, what else is happening around it). When in doubt, prefer Unclassified over a confident-looking wrong guess.

## Ledger heads and typical signals

**Revenue / Sales**
Inward credits that recur, come from many different counterparties, or are described as sale proceeds, invoice payments, or customer names/UPI handles matching a pattern of regular business income. `NEFT-CR-`, `IMPS`, `UPI-CR` from varied payer names, especially if amounts cluster around round invoice-like figures.

**Purchases / Cost of Goods Sold**
Outward payments to suppliers/vendors for goods resold or consumed in production. Narrations mentioning vendor/supplier names the business clearly deals with repeatedly, or explicit terms like "purchase," "supplier," "vendor payment."

**Direct expenses**
Costs directly tied to producing/delivering the goods or service sold — freight, carriage inward, direct wages/labour, packing. Only classify here (vs. indirect) if the business has a Trading Account; otherwise fold into indirect expenses.

**Indirect / operating expenses**
The bulk of recurring outflows: rent, salaries (`SALARY`, `SAL-`, staff names paid monthly), utilities (electricity, water, internet, phone), insurance premiums, professional fees (CA/legal/consulting), repairs & maintenance, office supplies, advertising/marketing, subscriptions. Bank charges and account maintenance fees (`SMS CHG`, `AMC`, `ANNUAL FEE`, `MIN BAL CHG`) always go here.

**Interest paid / received**
Interest debited by the bank on a loan/overdraft, or interest credited on a deposit — usually labeled explicitly (`INT.DEBIT`, `INT PAID`, `INTEREST CR`). Keep separate from principal loan movements.

**Capital introduced**
Owner's own money brought into the business — a personal-looking inward transfer with no invoice/customer pattern, especially early in the period or irregular in timing, particularly if it matches something the user confirms. When genuinely unsure whether an inward transfer is revenue or capital introduced, this is one of the highest-stakes flags to raise, since it changes both P&L and equity.

**Drawings**
Owner's personal withdrawals — cash withdrawals, transfers to a personal account, or payments for clearly personal expenses (household, personal shopping, family) made from the business account. For a sole proprietorship or partnership, personal-looking spending from the business account defaults to drawings unless narration or context says it's a business expense.

**Loan proceeds**
Inward lump sum from a bank/NBFC/lender, often with a loan account reference number in the narration, distinct from routine customer receipts by size and one-off timing.

**Loan repayment (principal)**
Outward EMI-type payments to a lender — split the interest portion (see above) from principal if the statement or an amortization schedule makes that possible; if not separable, flag it rather than lumping the whole EMI into either expenses or the balance sheet.

**Fixed asset purchase**
One-off, larger outward payments for equipment, vehicles, furniture, computers — capital expenditure, not an expense. Narration mentioning a dealer/showroom/asset description, or a payment size that's out of line with routine operating costs, is a signal to check rather than default-classify as an expense.

**Statutory payment**
GST, TDS, advance tax, PF/ESI — payments to government/statutory bodies. These settle liabilities already recognized elsewhere (or should be tracked as such) rather than being a fresh P&L expense in the period paid, unless the accounting basis is pure cash basis and the user wants it treated that way — ask if unclear.

**Internal transfer**
Movement between the business's own accounts (see main SKILL.md step 3). Excluded from P&L and revenue/expense totals entirely.

**Unclassified**
Anything that doesn't confidently fit above. Always record your best guess alongside the Unclassified tag so the user has something to react to rather than a blank.

## Worked examples

| Narration | Amount/context | Classification | Why |
|---|---|---|---|
| `UPI-RAJESH KUMAR-8827...-PAYMENT` | ₹45,000 credit, recurs monthly-ish | Revenue | Recurring inward from varied names, invoice-sized |
| `NEFT SALARY-OFFICE STAFF` | ₹18,000 debit, monthly | Indirect expense (salary) | Explicit label, recurring |
| `ACH DR-XYZ FINANCE-EMI` | ₹12,340 debit, monthly | Loan repayment (mixed principal/interest) | EMI pattern; flag to split if amortization unknown |
| `UPI-SELF TRANSFER` or same amount both accounts, same day | — | Internal transfer | Matches step-3 same-business-account rule |
| `CASH WDL ATM` | ₹10,000 | Drawings (default) unless business explains otherwise | No business narration attached |
| `NEFT-CR-XYZ CAPITAL LTD-LOAN A/C 88213` | ₹500,000, one-off, early in period | Loan proceeds | Explicit loan reference, one-off size |
| `UPI-SWIGGY` | ₹450, occasional | Unclassified (likely drawings, could be business meal) | Genuinely ambiguous — flag |