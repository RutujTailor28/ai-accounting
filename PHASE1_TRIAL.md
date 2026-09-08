# Phase 1 — Deterministic pipeline + public trial route

A bank statement in, a cash-basis balance sheet out, with **no LLM anywhere**.
Built as a self-contained vertical slice so it cannot affect the existing
production pipeline.

## What it does

```
upload -> parse tables -> validate running balance -> de-duplicate
       -> normalise narrations -> seed rules -> journal -> ledgers
       -> trial balance -> P&L -> balance sheet
```

Every stage is pure Python and `Decimal`. The same file always produces the
same statements. A 2,000-row statement completes in well under a second,
which is why the route is synchronous — **no queue, no worker, and no
serverless timeout problem.**

## Honest limits — say these to the accountants

- It is a **cash-basis summary from bank movement only**. No debtors,
  creditors, stock, fixed assets or depreciation, because a bank statement
  cannot evidence them. It is not a statutory balance sheet.
- "The balance sheet balances" proves nothing about accuracy. Both sides are
  equal *by construction*; rent booked as salary still balances. The real
  test is an accountant comparing it against their own figures.
- Unmatched debits go to **Suspense**, never to a guessed ledger.

## Endpoints (`/api/v1/trial`, no auth)

| Method | Path | Purpose |
|---|---|---|
| POST | `/session` | Exchange an access code for a session id |
| GET | `/chart` | Ledger heads for the reclassify dropdown |
| POST | `/analyze` | Upload a statement, get statements back |
| POST | `/reclassify` | Change ledgers, rebuild the arithmetic |
| POST | `/feedback` | Record the accountant's verdict |

Only **parsed transactions** are stored — never the uploaded file.
Reclassification re-runs from those, so a correction can never be lost to a
different parse of the same file.

## Guardrails

Public, so it is deliberately fenced:

- Shared access code (`TRIAL_ACCESS_CODES`) — not accounts, just enough to
  keep the open internet out.
- 10 MB and 5 files per session; 10 sessions per IP per day.
- Sessions expire after 72 hours.
- A public endpoint that called a model would be an uncapped bill. This one
  cannot: there is a test asserting the package imports no model client.

## Setup

```bash
psql "$DATABASE_URL" -f migrations/002_trial.sql   # or paste into Supabase SQL editor
export TRIAL_ACCESS_CODES="PILOT-2026,FIRM-A"
```

Without the migration it falls back to per-process memory — fine on a dev
machine, useless on serverless. **Apply it before sending the link.**

Then share: `https://<your-app>/trial?code=PILOT-2026`

## Tests

```bash
python tests/test_phase1.py     # 47 checks, no network
```

Covers the balance gate rejecting a corrupted statement, duplicate payments
surviving de-duplication, `Decimal` precision, reviewer overrides, the
normaliser, and a guard asserting no model imports.

## Files

| Path | Purpose |
|---|---|
| `app/phase1/contracts.py` | Canonical `Transaction`, money handling |
| `app/phase1/coa.py` | Chart of accounts |
| `app/phase1/normalise.py` | Narration -> counterparty + channel |
| `app/phase1/rules.py` | Seed classification rules |
| `app/phase1/parsers.py` | Table extraction + running-balance gate |
| `app/phase1/ledger.py` | Journal -> ledgers -> trial balance |
| `app/phase1/statements.py` | P&L + balance sheet |
| `app/phase1/pipeline.py` | Orchestrator |
| `app/phase1/trial_store.py` | Session/document/feedback storage |
| `app/api/v1/endpoints/trial.py` | The public route |
| `migrations/002_trial.sql` | Tables + `trial_feedback_report` view |

Frontend page: `app/trial/page.tsx` in the front-end repo.

## Reading the feedback

```sql
select * from trial_feedback_report;
```

One row per submission with the filename, verdict, what was wrong, and the
auto-classification coverage for that file.
