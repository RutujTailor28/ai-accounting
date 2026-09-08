-- Public trial route (Phase 1 pilot).
-- Only parsed transactions are stored - never the uploaded file.

create table if not exists public.trial_sessions (
    id            text primary key,
    access_code   text not null,
    ip            text,
    user_agent    text,
    upload_count  integer     not null default 0,
    created_at    timestamptz not null default now(),
    expires_at    timestamptz not null
);

create table if not exists public.trial_documents (
    id              text primary key,
    session_id      text not null references public.trial_sessions (id) on delete cascade,
    filename        text not null,
    transactions    jsonb not null default '[]'::jsonb,
    result          jsonb not null default '{}'::jsonb,
    opening_balance text,
    created_at      timestamptz not null default now()
);

create table if not exists public.trial_feedback (
    id         bigserial primary key,
    session_id text not null references public.trial_sessions (id) on delete cascade,
    payload    jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists trial_documents_session_idx on public.trial_documents (session_id);
create index if not exists trial_feedback_session_idx  on public.trial_feedback (session_id);
create index if not exists trial_sessions_expires_idx  on public.trial_sessions (expires_at);

-- Written only by the backend service role; no public policy on purpose.
alter table public.trial_sessions  enable row level security;
alter table public.trial_documents enable row level security;
alter table public.trial_feedback  enable row level security;

-- All feedback with the file it refers to, newest first.
create or replace view public.trial_feedback_report as
select
    f.created_at,
    f.session_id,
    d.filename,
    f.payload ->> 'verdict'         as verdict,
    f.payload ->> 'would_use'       as would_use,
    f.payload ->> 'what_is_wrong'   as what_is_wrong,
    f.payload ->> 'misclassified_rows' as misclassified_rows,
    f.payload ->> 'contact'         as contact,
    d.result -> 'coverage'          as coverage
from public.trial_feedback f
left join public.trial_documents d
       on d.id = f.payload ->> 'document_id'
order by f.created_at desc;

-- Purge expired trial data. Run on a schedule, or by hand after the pilot.
-- delete from public.trial_sessions where expires_at < now();
