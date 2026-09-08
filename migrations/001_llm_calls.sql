-- Phase 0 instrumentation: per-call LLM cost telemetry.
-- Optional. The JSONL log at logs/llm_calls.jsonl works without this table.
-- Apply this, then set LLM_TELEMETRY_SUPABASE=true to also persist to Postgres.

create table if not exists public.llm_calls (
    id               bigserial primary key,
    run_id           text        not null,
    call_id          text        not null,
    ts               timestamptz not null default now(),
    stage            text        not null,
    model            text        not null,
    provider         text,
    input_tokens     integer     not null default 0,
    output_tokens    integer     not null default 0,
    cached_tokens    integer     not null default 0,
    total_tokens     integer     not null default 0,
    cost_usd         numeric(12, 8),
    -- false => no price configured for this model; cost_usd is meaningless
    cost_known       boolean     not null default false,
    -- true => provider returned no usage block; token counts are approximate
    estimated        boolean     not null default false,
    latency_ms       integer     not null default 0,
    ok               boolean     not null default true,
    error            text,
    prompt_chars     integer     not null default 0,
    completion_chars integer     not null default 0,
    client_id        text,
    created_at       timestamptz not null default now()
);

create index if not exists llm_calls_run_id_idx  on public.llm_calls (run_id);
create index if not exists llm_calls_stage_idx   on public.llm_calls (stage);
create index if not exists llm_calls_ts_idx      on public.llm_calls (ts desc);
create index if not exists llm_calls_client_idx  on public.llm_calls (client_id);

-- Written only by the backend service role, so keep RLS on with no public policy.
alter table public.llm_calls enable row level security;

-- Cost per run, most expensive first.
create or replace view public.llm_run_costs as
select
    run_id,
    min(ts)                                   as started_at,
    max(client_id)                            as client_id,
    count(*)                                  as calls,
    sum(input_tokens)                         as input_tokens,
    sum(output_tokens)                        as output_tokens,
    sum(cached_tokens)                        as cached_tokens,
    sum(cost_usd) filter (where cost_known)   as cost_usd,
    bool_and(cost_known)                      as cost_complete,
    bool_or(estimated)                        as tokens_estimated,
    count(*) filter (where not ok)            as failed_calls
from public.llm_calls
group by run_id
order by cost_usd desc nulls last;

-- Which pipeline stage burns the tokens.
create or replace view public.llm_stage_costs as
select
    stage,
    count(*)                                                as calls,
    sum(input_tokens)                                       as input_tokens,
    sum(output_tokens)                                      as output_tokens,
    sum(cost_usd) filter (where cost_known)                 as cost_usd,
    round(avg(latency_ms))                                  as avg_latency_ms,
    round(100.0 * sum(input_tokens + output_tokens)
          / nullif(sum(sum(input_tokens + output_tokens)) over (), 0), 1) as pct_tokens
from public.llm_calls
group by stage
order by sum(input_tokens + output_tokens) desc;
