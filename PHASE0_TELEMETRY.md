# Phase 0 — LLM Cost Instrumentation

Answers one question: **where does the money actually go?**

Nothing here changes prompts, models, temperature, or control flow. Every
telemetry path swallows its own errors, so a bug in instrumentation cannot
break a generation run. To switch it off entirely: `LLM_TELEMETRY_ENABLED=false`.

## How to use it

1. Run a normal accounting generation from the UI or API.
2. A cost table prints at the end of the run.
3. Aggregate across many runs:

```bash
python -m app.core.llm_telemetry logs/llm_calls.jsonl
```

Example:

```
BY STAGE                        CALLS        IN      OUT    COST $   % TOK
AGENT-1 Extractor                  34   612,000   18,400   0.63040   71.2%
AGENT-2 Classifier                  9    41,000    7,500   0.07850   12.4%
AGENT-5 Balance Sheet               1     1,200      600   0.00420    1.1%
```

The `% TOK` column is the answer. Whatever sits at the top is where the
budget goes and what Phase 1 must replace first.

## What gets recorded

One row per model call, in `logs/llm_calls.jsonl` (gitignored):

| field | meaning |
|---|---|
| `stage` | which agent made the call (`AGENT-2 Classifier`, …) |
| `input_tokens` / `output_tokens` | from the provider's usage block |
| `cached_tokens` | prompt-cache hits, when the provider reports them |
| `cost_usd` / `cost_known` | `cost_known: false` ⇒ no price configured; cost is **not** zero, it is unknown |
| `estimated` | `true` ⇒ provider returned no usage block, counts are approximate |
| `latency_ms`, `ok`, `error` | failed calls are recorded too — they still burn input tokens |

## Pricing

Costs only appear for models priced in `config/llm_pricing.json`:

```json
{ "models": { "meta/llama-3.1-8b-instruct": {
    "input_per_mtok": 0.20, "output_per_mtok": 0.20 } } }
```

Unpriced models report **tokens only** and print a warning naming them,
rather than inventing a number. Models ending in `:free` are treated as zero.
Token counts are the reliable signal; dollars depend on this file being right.

## Two things that silently break token counts

- **`streaming=True` hides usage.** `AccountingService` streams, which
  suppresses the usage block unless `stream_options.include_usage` is set.
  That is why the client is now built with `stream_usage=True`. If your
  provider rejects that parameter, set `LLM_TELEMETRY_STREAM_USAGE=false`;
  counts then fall back to estimates and are flagged `estimated: true`.
- **Estimates are ~chars/4**, never presented as exact. If a whole run is
  flagged estimated, the provider is not returning usage — fix that before
  trusting any number.

## Optional: persist to Postgres

The JSONL log is enough for Phase 0. For dashboards:

```bash
psql "$DATABASE_URL" -f migrations/001_llm_calls.sql   # or paste into Supabase SQL editor
export LLM_TELEMETRY_SUPABASE=true
```

Adds the `llm_calls` table plus two views: `llm_run_costs` (cost per run,
most expensive first) and `llm_stage_costs` (which stage burns tokens).

## Where the wiring lives

| file | change |
|---|---|
| `app/core/llm_telemetry.py` | the whole module (new) |
| `app/services/ai_service.py` | attaches callbacks to `ChatOpenAI` |
| `app/services/accounting_service.py` | callbacks + `stream_usage`, `@instrument_stream`, 9 `llm_stage(...)` labels |
| `config/llm_pricing.json` | per-model prices (fill in) |
| `migrations/001_llm_calls.sql` | optional table + views |

Coverage is by callback on the two `ChatOpenAI` instances, so **every** call
is captured regardless of call path. Calls outside a labelled stage appear as
`unattributed` — if that row is large, a call site is missing a label.
