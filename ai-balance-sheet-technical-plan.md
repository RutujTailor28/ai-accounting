# AI Balance Sheet Pipeline — Technical Plan

**Goal:** Generate a balance sheet (and P&L / trial balance) for one client from bank statements and supporting documents at a marginal LLM cost of **under $0.10 per client**, with accuracy an accountant will trust.

**Target economics:** $20/month plan → 50 clients/month → hard budget of $0.40 per client, design target $0.10.

**Core principle:** The LLM classifies transactions. Code does everything else (parsing, ledgers, arithmetic, statements). Never send a whole statement to a model, never ask a model to do arithmetic, never let a model echo data back.

---

## 1. Architecture Overview

```
Upload (PDF/XLSX/CSV)
   │
   ▼
[1] Document Parser  ──►  transactions table (structured rows)
   │
   ▼
[2] Normaliser       ──►  cleaned narration, counterparty key, txn_type
   │
   ▼
[3] Rules Engine     ──►  ~70–90% rows classified (no LLM)
   │      │
   │      └─ unmatched rows
   ▼
[4] LLM Classifier   ──►  batches of 50–100 rows → JSON {row_id, ledger, confidence}
   │      │
   │      └─ low confidence → escalation model or human review queue
   ▼
[5] Accountant Review UI (optional per run) ──► corrections feed back into rules memory
   │
   ▼
[6] Ledger Engine    ──►  journal → ledgers → trial balance (pure code)
   │
   ▼
[7] Statement Builder ──► Balance Sheet, P&L (Schedule III or simple format) → PDF/XLSX
   │
   ▼
[8] AI Review        ──►  one small LLM call on a compact summary → anomaly notes
```

**Stack recommendation** (adjust to what your team knows):
- Backend: Python (FastAPI). Python has the best PDF/table tooling.
- DB: PostgreSQL. Drop the vector DB for this feature; it's not needed.
- Queue: Celery + Redis (or RQ). Balance sheet generation runs as a background job.
- LLM: call the provider directly (Anthropic / Google / OpenAI), not via OpenRouter, once a model is chosen. Use structured output / JSON mode.
- Statement output: openpyxl for XLSX, WeasyPrint or ReportLab for PDF.

---

## 2. Data Model (PostgreSQL)

```sql
-- One row per accountant (your paying user)
accountants (id, name, email, plan, created_at)

-- One row per accountant's client
clients (
  id, accountant_id, name, pan, gstin, entity_type,   -- proprietorship | partnership | pvt_ltd | llp
  financial_year_start DATE, opening_balances JSONB, created_at
)

-- Chart of accounts. Seed a default per entity_type; accountant can customise.
ledger_heads (
  id, client_id NULL,          -- NULL = global default
  code TEXT,                   -- e.g. "5010"
  name TEXT,                   -- "Rent Expense"
  group_name TEXT,             -- "Indirect Expenses"
  bs_section TEXT,             -- "Current Liabilities" | "Fixed Assets" | NULL for P&L
  pl_section TEXT,             -- "Direct Expenses" | "Indirect Income" | NULL for BS
  nature TEXT                  -- asset | liability | equity | income | expense
)

-- Uploaded files
documents (id, client_id, type, file_path, bank_name, account_no, period_from, period_to,
           parse_status, parsed_at, row_count)

-- The central table
transactions (
  id, client_id, document_id,
  txn_date DATE, narration_raw TEXT, narration_clean TEXT,
  debit NUMERIC(14,2), credit NUMERIC(14,2), balance NUMERIC(14,2),
  ref_no TEXT, channel TEXT,          -- UPI | NEFT | IMPS | RTGS | CHQ | ATM | CHARGES | INT | OTHER
  counterparty_key TEXT,              -- normalised, used for rule matching
  ledger_head_id INT NULL,
  classification_source TEXT,         -- rule | memory | llm | llm_escalated | human
  confidence NUMERIC(3,2),
  needs_review BOOLEAN DEFAULT false,
  fingerprint TEXT UNIQUE             -- sha256(client_id, date, amount, narration_raw) for dedup
)

-- Learned mappings. This is what makes the system get cheaper every month.
classification_memory (
  id, scope TEXT,                     -- client | accountant | global
  scope_id INT,
  counterparty_key TEXT,
  ledger_head_id INT,
  hit_count INT, last_used_at, created_by TEXT  -- human | llm
)

-- Static rules
classification_rules (
  id, scope, scope_id, priority INT,
  pattern TEXT, pattern_type TEXT,    -- regex | contains | channel | amount_range
  direction TEXT,                     -- debit | credit | any
  ledger_head_id INT, active BOOLEAN
)

-- Job tracking + cost telemetry (mandatory)
generation_runs (
  id, client_id, fy, status, started_at, finished_at,
  rows_total, rows_rule, rows_memory, rows_llm, rows_escalated, rows_review,
  llm_input_tokens, llm_output_tokens, llm_cached_tokens, llm_cost_usd,
  output_paths JSONB
)

llm_calls (id, run_id, model, purpose, input_tokens, output_tokens, cached_tokens,
           cost_usd, latency_ms, batch_size, created_at)
```

---

## 3. Stage-by-Stage Specification

### Stage 1 — Document Parser

**Input:** bank statement PDF / XLSX / CSV. Later: sales/purchase registers, GST returns (GSTR-1/3B), Form 26AS.

**Approach:**
1. XLSX/CSV: load with pandas, detect header row by scanning for column names matching `date|narration|particulars|description|withdrawal|debit|deposit|credit|balance`.
2. PDF text-based: `pdfplumber` table extraction. Write one parser adapter per bank format (SBI, HDFC, ICICI, Axis, Kotak, BoB, PNB cover most SMEs). Each adapter maps columns → the canonical transaction schema.
3. PDF scanned: OCR with Tesseract or a cloud OCR (Google Document AI / Azure Form Recognizer). OCR is a fixed per-page cost, usually cheaper than LLM parsing. **Do not use an LLM to parse a whole statement**; only fall back to a small vision model for pages where table extraction fails validation.
4. **Validation gate:** for every parsed statement, verify `running_balance[i] == running_balance[i-1] - debit + credit`. If more than 2% of rows fail, flag the document as "parse failed, needs manual check." This catches OCR errors before they poison the ledger.
5. Dedup by `fingerprint`. Statements often overlap across uploads.

**Output:** rows in `transactions` with `ledger_head_id = NULL`.

### Stage 2 — Normaliser

Pure Python, no LLM.

```python
def normalise(narration: str) -> tuple[str, str, str]:
    # returns (narration_clean, counterparty_key, channel)
```

- Detect channel from prefixes: `UPI/`, `NEFT`, `IMPS`, `RTGS`, `CHQ`, `ATM`, `POS`, `INT.PD`, `CHRG`, `SMS CHG`, etc.
- Strip transaction IDs, dates, reference numbers, UTR numbers (long digit runs).
- Extract counterparty: for UPI take the VPA (`name@bank`) or the name segment; for NEFT/IMPS take the beneficiary name segment; for cheques take the payee if present.
- `counterparty_key = lowercase, alphanumerics only, collapse whitespace`. Example: `UPI/423122/PAYMENT/SWIGGY@AXISBANK/…` → key `swiggy`.

Test this heavily with real narrations from each bank. This function determines the hit rate of Stage 3.

### Stage 3 — Rules Engine

Runs in priority order; first match wins.

1. **Client memory:** `classification_memory` where `scope=client`. Exact match on `counterparty_key` + direction.
2. **Accountant memory:** `scope=accountant`. The same vendor across many of this accountant's clients.
3. **Global memory:** `scope=global`. Well-known counterparties (GST portal, income tax, EPFO, ESIC, common SaaS vendors, telecom, utilities).
4. **Static rules:** regex/contains patterns. Seed with ~100 rules, e.g.
   - `channel=INT & credit` → Interest Income (Bank)
   - `channel=CHARGES` → Bank Charges
   - `GST|CBIC|GSTN` debit → GST Payable / Duties & Taxes
   - `TDS|INCOME TAX|ITNS` debit → TDS Payable or Advance Tax (depends; mark medium confidence)
   - `EMI|LOAN` debit → Loan repayment (split principal/interest is a later feature; initially book to loan account)
   - `SALARY|SAL ` debit → Salaries
   - `ATM|CASH WDL` debit → Cash
   - Transfers to/from the client's own other accounts (match account numbers on file) → Contra
5. **Recurrence heuristic:** if the same `counterparty_key` appears ≥3 times with the same direction and one of them is already classified (any source), copy that classification with confidence 0.85.

Everything matched here has `classification_source ∈ {rule, memory}` and never touches an LLM.

**Expected outcome:** 60–75% coverage on a brand-new client, 85–95% on a returning client after the accountant's first review.

### Stage 4 — LLM Classifier

Only unmatched rows reach here.

**Batching:** group unmatched rows into batches of 60 (tune 50–100). Sort by counterparty_key so similar rows sit together; the model classifies them consistently.

**Prompt structure** (keep the fixed part identical across calls so it is cache-hit):

```
[SYSTEM — cached]
You are classifying bank transactions for an Indian {entity_type} into ledger heads.
Chart of accounts (code: name — when to use):
5010: Rent Expense — office/shop rent, ...
... (full CoA, ~60–120 lines)
Rules:
- Debits to individuals with no other context → "Suspense (Review)" with confidence ≤ 0.5
- ...
Return ONLY JSON matching the schema. Never explain.

[USER — per batch]
Client context: {business_description, e.g. "retail pharmacy, Ahmedabad"}
Rows:
r1|2025-04-03|D|12500.00|UPI|rajesh traders|UPI/…/RAJESH TRADERS/
r2|2025-04-03|C|8000.00|UPI|customer paytm|…
...
```

**Output schema (enforced via structured output / tool call):**
```json
{"rows":[{"id":"r1","ledger":"5210","confidence":0.92},{"id":"r2","ledger":"4010","confidence":0.71}]}
```

Compact row format (pipe-delimited, no JSON on input) keeps input at ~25–40 tokens per row. Output is ~15 tokens per row.

**Model tiering:**
- Tier 1 (default): smallest capable model — Claude Haiku, Gemini Flash, GPT-4o-mini class, or a hosted open model (Qwen/Llama 8–70B). Pick by running the accuracy benchmark in §6.
- Tier 2 (escalation): rows with confidence < 0.6 from Tier 1 are re-sent, in one batch, to a mid-tier model (Sonnet class). Expect < 10% of LLM rows to escalate.
- Rows still < 0.6 after Tier 2 → `needs_review = true`, booked to "Suspense (Review)".

**Cost controls (implement all):**
- Prompt caching on the system block (Anthropic `cache_control`, OpenAI automatic caching, Gemini context caching).
- Use the provider's **Batch API** for generation runs (≈50% discount). Runs are async anyway.
- `max_tokens` = `batch_size * 25 + 50`. Hard cap.
- Retry with exponential backoff; on malformed JSON, retry once, then mark batch for review. Never loop indefinitely.
- Every call writes a row to `llm_calls` with token counts and computed cost. A run aborts if `llm_cost_usd > 0.50` (configurable circuit breaker).

**Write-back:** every LLM classification with confidence ≥ 0.85 is written to `classification_memory` with `scope=client, created_by=llm`. Human corrections overwrite it and also promote to `scope=accountant` if the same key exists across ≥ 2 clients.

### Stage 5 — Accountant Review UI

- Show rows with `needs_review = true` first, then low-confidence rows, then everything grouped by ledger.
- Bulk actions: "apply this ledger to all rows with the same counterparty."
- Every correction → `classification_memory` (human) and, if the accountant ticks "always," → `classification_rules`.
- Show a coverage summary: "412 transactions: 380 auto-classified, 24 by AI, 8 need your review."

This UI is what makes month two cheaper than month one.

### Stage 6 — Ledger Engine (pure code)

```python
class Journal:
    def add(self, date, debit_ledger, credit_ledger, amount, txn_id)

def build_journal(client, transactions, opening_balances) -> Journal
def build_ledgers(journal) -> dict[ledger_code, LedgerAccount]   # running balances
def trial_balance(ledgers) -> TrialBalance                         # must sum to zero
```

- Each bank transaction becomes one journal entry: bank ledger on one side, classified ledger on the other. Bank account itself is a ledger head (asset).
- Load opening balances from `clients.opening_balances` (previous year's closing, or entered manually).
- Financial year default: 1 April – 31 March. Filter transactions to FY.
- **Assertion:** `sum(debits) == sum(credits)` in the trial balance. If not, the run fails loudly. Use `Decimal`, never float.
- Adjustments layer (v2): depreciation, closing stock, accruals, loan interest split. Store as manual journal entries in a `journal_adjustments` table so they survive re-runs.

### Stage 7 — Statement Builder

- Map `ledger_heads.bs_section` / `pl_section` into the statement layout.
- P&L: Income − Expenses = Net Profit → transferred to Capital / Reserves & Surplus.
- Balance Sheet: Assets = Liabilities + Equity. **Assert equality**; fail the run if it doesn't tie.
- Output formats: XLSX (with formulas so the accountant can trace), PDF, and JSON (for the UI).
- Formats: simple vertical format for proprietorship/partnership; Schedule III format for companies (v2).

### Stage 8 — AI Review Call (one small call)

Input: trial balance (≤ 100 lines), P&L totals, top 10 ledgers by movement, count of suspense rows, YoY deltas if available. ~2–3k tokens.
Prompt: "List up to 8 observations an accountant should check before filing." Output ≤ 400 tokens.
Model: Tier 1. This is a nice-to-have; keep it behind a feature flag.

---

## 4. Cost Model (design target)

Assume a typical SME client: 2,000 transactions/year.

| Step | Rows | Tokens in | Tokens out | Notes |
|---|---|---|---|---|
| Rules/memory | 1,500 | 0 | 0 | free |
| Tier 1 LLM | 500 | ~18k (mostly cached) | ~8k | 9 batches × 60 |
| Tier 2 escalation | 40 | ~3k | ~700 | 1 batch |
| AI review | – | ~3k | ~400 | optional |

At small-model pricing this lands around **$0.01–0.04 per client**, with batch discount roughly halving it. Even a worst-case brand-new client with 0% rule coverage (2,000 rows to Tier 1) stays under ~$0.10. Verify against current provider pricing before finalising the model choice; prices change often.

Compare with today: ~$20 ≈ 1–3M frontier-model tokens, i.e. the whole statement re-sent many times.

---

## 5. Build Phases

**Phase 0 — Instrumentation (1–2 days).** Before changing anything, log tokens and cost per LLM call in the current system. Get the actual breakdown of the $20. This confirms the diagnosis and gives a baseline.

**Phase 1 — Parser + Normaliser + Ledger Engine (2–3 weeks).**
Parsers for the 5–7 most common banks. Normaliser. Ledger engine and statement builder. At the end of this phase you can produce a balance sheet from a statement where every row is manually classified. Zero LLM.

**Phase 2 — Rules Engine + Memory (1–2 weeks).**
Seed rules, memory tables, review UI. Measure coverage on real client data.

**Phase 3 — LLM Classifier (1–2 weeks).**
Batching, structured output, caching, batch API, tiering, circuit breaker, cost telemetry. Run the model benchmark (§6) and pick Tier 1.

**Phase 4 — Hardening (1–2 weeks).**
More bank parsers, OCR fallback, opening balances, adjustments, PDF/XLSX export polish, AI review call.

**Phase 5 (later) — Fine-tune.** Once you have ~5–10k human-verified classifications, fine-tune a small open model or use a provider's fine-tuning. Expect higher accuracy than general models at lower cost.

---

## 6. Testing Plan

### 6.1 Unit tests
- **Normaliser:** table-driven tests, ≥ 200 real narrations per bank → expected `(channel, counterparty_key)`. Target 98% pass.
- **Parser per bank:** golden files. Parse sample PDF/XLSX → compare against hand-verified CSV. Running-balance validation must pass 100% on clean statements.
- **Ledger engine:** synthetic journals with known outcomes. Trial balance always zero. Balance sheet always ties. Test Decimal rounding, FY boundaries, opening balances, contra entries.
- **Rules engine:** each seeded rule has ≥ 3 positive and 3 negative narration examples.
- **LLM output parsing:** malformed JSON, missing row ids, unknown ledger codes, duplicate ids → all handled without crashing.

### 6.2 Classification accuracy benchmark
- Build a **gold dataset**: 3–5 real clients (anonymised), every transaction labelled by an accountant. Aim for 3,000+ rows across entity types.
- Metrics: accuracy per source (rule / memory / LLM tier 1 / tier 2), coverage (% not needing review), and cost per client.
- Run the benchmark script against each candidate Tier 1 model. Pick the cheapest model with ≥ 90% accuracy on rows it marks confidence ≥ 0.85.
- Re-run the benchmark in CI whenever the prompt, CoA, or model changes. Fail CI if accuracy drops > 2 points.

### 6.3 Cost regression tests
- Fixture client with 2,000 rows and empty memory (worst case). Assert `run.llm_cost_usd < 0.15`.
- Same client, second run after memory populated. Assert `rows_llm < 300` and cost `< 0.05`.
- Assert cache hit: `llm_cached_tokens / llm_input_tokens > 0.6` on runs with ≥ 3 batches.
- Circuit breaker test: mock a pricing spike and confirm the run aborts.

### 6.4 End-to-end
- Upload statement → generate → download XLSX/PDF. Compare generated trial balance to the accountant's own Tally/Excel output for the gold clients. Differences should be explainable (only suspense/review rows).
- Load test: 50 clients queued at once; all complete; batch API used; total cost logged.

### 6.5 Acceptance criteria for launch
- Balance sheet ties for 100% of runs (or run fails with a clear error).
- Median cost per client < $0.10; p95 < $0.30.
- Auto-classification coverage ≥ 85% on returning clients.
- Accountant review time < 10 minutes for a 2,000-row client (measure with 3 pilot users).

---

## 7. Things to Explicitly Stop Doing

- Sending entire statements or retrieved vector chunks to the model.
- Multi-step agent loops that re-read the same context.
- Asking the model to produce ledgers, totals, or the balance sheet itself.
- Using a frontier model for routine classification.
- Calling the API without token/cost logging.

---

## 8. Open Decisions for the Team

1. Which 5–7 bank formats to support first (check what your pilot accountants' clients actually use).
2. Entity types for v1 (recommend proprietorship + partnership; add company/Schedule III in v2).
3. Tier 1 model — decide from the benchmark, not by reputation.
4. Whether the AI review call ships in v1 or v2.
5. OCR provider for scanned statements (cost per page vs accuracy).
