"""Phase 0 instrumentation: per-call LLM token and cost telemetry.

This module is deliberately *additive*. Enabling it must never change prompts,
model selection, temperature, or control flow. Every public entry point
swallows its own exceptions, so a bug in telemetry can never break a
generation run.

What it answers
---------------
"Where does the money actually go?" - broken down per pipeline stage
(Agent 1 extractor, Agent 2 classifier, Agent 4 P&L, ...), per model, with
token counts, latency and cost.

Wiring (already done in this repo)
----------------------------------
1. `LLMService.__init__` and `AccountingService.__init__` attach
   `telemetry_callbacks()` to their `ChatOpenAI` instances. That is what
   captures every call - no per-call-site changes needed.
2. Pipeline stages wrap their work in `with llm_stage("AGENT-2 Classifier"):`
   so calls are attributed rather than lumped into "unattributed".
3. A whole run is wrapped in `with llm_run(...)`, which prints a summary
   table when the run finishes.

Reading the output
------------------
    python -m app.core.llm_telemetry logs/llm_calls.jsonl

Pricing
-------
Costs are only reported for models that have a price configured in
`config/llm_pricing.json` (or the `LLM_PRICING_JSON` env var). Unpriced
models report tokens with `cost_known: false` rather than a fabricated
number - see `report_unpriced_models()`.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TELEMETRY_ENABLED = os.getenv("LLM_TELEMETRY_ENABLED", "true").lower() not in (
    "0",
    "false",
    "no",
)

# Where the per-call JSONL log is written. `logs/` is already gitignored.
TELEMETRY_LOG_PATH = Path(os.getenv("LLM_TELEMETRY_LOG", "logs/llm_calls.jsonl"))

# Persist each call to the Supabase `llm_calls` table as well. Off by default:
# turn on only after running migrations/001_llm_calls.sql.
TELEMETRY_TO_SUPABASE = os.getenv("LLM_TELEMETRY_SUPABASE", "false").lower() in (
    "1",
    "true",
    "yes",
)

_PRICING_PATH = Path(os.getenv("LLM_PRICING_FILE", "config/llm_pricing.json"))

# `ChatOpenAI(streaming=True)` omits the usage block unless the request also
# sets `stream_options={"include_usage": true}`. LangChain exposes that as
# `stream_usage`. If a provider rejects the extra parameter, set
# LLM_TELEMETRY_STREAM_USAGE=false - token counts then fall back to estimates
# (and are flagged `estimated: true` rather than reported as exact).
STREAM_USAGE = os.getenv("LLM_TELEMETRY_STREAM_USAGE", "true").lower() not in (
    "0",
    "false",
    "no",
)

# Rough characters-per-token used only when the provider returns no usage
# block. Anything derived from this is flagged `estimated: true`.
_CHARS_PER_TOKEN = 4.0

_write_lock = threading.Lock()


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens. `None` means "not configured" - never assume zero."""

    input_per_mtok: Optional[float] = None
    output_per_mtok: Optional[float] = None
    cached_input_per_mtok: Optional[float] = None


_pricing_cache: Optional[Dict[str, ModelPrice]] = None
_unpriced_models: set[str] = set()


def _normalise_model(name: Optional[str]) -> str:
    return (name or "unknown").strip().lower()


def load_pricing() -> Dict[str, ModelPrice]:
    """Load the model -> price map from env JSON or `config/llm_pricing.json`."""
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache

    raw: Dict[str, Any] = {}
    env_json = os.getenv("LLM_PRICING_JSON")
    if env_json:
        try:
            raw = json.loads(env_json)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[TELEMETRY][WARN] LLM_PRICING_JSON is not valid JSON: {exc}")
    elif _PRICING_PATH.exists():
        try:
            raw = json.loads(_PRICING_PATH.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[TELEMETRY][WARN] Could not read {_PRICING_PATH}: {exc}")

    prices: Dict[str, ModelPrice] = {}
    for model, entry in (raw.get("models") or {}).items():
        if not isinstance(entry, dict):
            continue
        prices[_normalise_model(model)] = ModelPrice(
            input_per_mtok=entry.get("input_per_mtok"),
            output_per_mtok=entry.get("output_per_mtok"),
            cached_input_per_mtok=entry.get("cached_input_per_mtok"),
        )

    _pricing_cache = prices
    return prices


def price_for(model: str) -> ModelPrice:
    """Resolve a price, falling back to a zero price for `:free` models."""
    prices = load_pricing()
    key = _normalise_model(model)
    if key in prices:
        return prices[key]
    # OpenRouter ":free" variants are unambiguously zero-cost.
    if key.endswith(":free"):
        return ModelPrice(0.0, 0.0, 0.0)
    _unpriced_models.add(key)
    return ModelPrice()


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> tuple[Optional[float], bool]:
    """Return `(cost_usd, cost_known)`.

    OpenAI-compatible providers report `prompt_tokens` *inclusive* of cached
    tokens, so the fresh (fully billed) portion is the difference.
    """
    p = price_for(model)
    if p.input_per_mtok is None or p.output_per_mtok is None:
        return None, False

    fresh_input = max(0, input_tokens - max(0, cached_tokens))
    cached_rate = (
        p.cached_input_per_mtok
        if p.cached_input_per_mtok is not None
        else p.input_per_mtok
    )

    cost = (
        fresh_input * p.input_per_mtok
        + max(0, cached_tokens) * cached_rate
        + output_tokens * p.output_per_mtok
    ) / 1_000_000.0
    return cost, True


def report_unpriced_models() -> List[str]:
    """Models seen this process that have no configured price."""
    return sorted(_unpriced_models)


# --------------------------------------------------------------------------
# Call + run records
# --------------------------------------------------------------------------


@dataclass
class LLMCall:
    """One model invocation."""

    run_id: str
    call_id: str
    ts: str
    stage: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    total_tokens: int
    cost_usd: Optional[float]
    cost_known: bool
    estimated: bool
    latency_ms: int
    ok: bool
    error: Optional[str] = None
    prompt_chars: int = 0
    completion_chars: int = 0
    client_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class RunAccumulator:
    """Collects every call made inside one `llm_run(...)` block."""

    run_id: str
    label: str
    client_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    calls: List[LLMCall] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, call: LLMCall) -> None:
        with self._lock:
            self.calls.append(call)

    # -- aggregates -------------------------------------------------------

    @property
    def total_input(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_cached(self) -> int:
        return sum(c.cached_tokens for c in self.calls)

    @property
    def total_cost(self) -> Optional[float]:
        known = [c.cost_usd for c in self.calls if c.cost_known and c.cost_usd is not None]
        if not known:
            return None
        return sum(known)

    @property
    def any_cost_unknown(self) -> bool:
        return any(not c.cost_known for c in self.calls)

    @property
    def any_estimated(self) -> bool:
        return any(c.estimated for c in self.calls)

    def by_stage(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for c in self.calls:
            row = out.setdefault(
                c.stage,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "cost_usd": 0.0,
                    "cost_known": True,
                    "errors": 0,
                    "latency_ms": 0,
                },
            )
            row["calls"] += 1
            row["input_tokens"] += c.input_tokens
            row["output_tokens"] += c.output_tokens
            row["cached_tokens"] += c.cached_tokens
            row["latency_ms"] += c.latency_ms
            if not c.ok:
                row["errors"] += 1
            if c.cost_known and c.cost_usd is not None:
                row["cost_usd"] += c.cost_usd
            else:
                row["cost_known"] = False
        return out

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "client_id": self.client_id,
            "duration_s": round(time.time() - self.started_at, 2),
            "calls": len(self.calls),
            "input_tokens": self.total_input,
            "output_tokens": self.total_output,
            "cached_tokens": self.total_cached,
            "total_tokens": self.total_input + self.total_output,
            "cost_usd": self.total_cost,
            "cost_complete": not self.any_cost_unknown,
            "tokens_estimated": self.any_estimated,
            "by_stage": self.by_stage(),
        }

    def format_summary(self) -> str:
        """Human-readable breakdown - this is the Phase 0 deliverable."""
        stages = self.by_stage()
        total_tokens = self.total_input + self.total_output

        lines: List[str] = []
        lines.append("=" * 78)
        lines.append(f"[TELEMETRY] LLM cost report - run {self.run_id} ({self.label})")
        lines.append("=" * 78)
        lines.append(
            f"{'STAGE':<32}{'CALLS':>6}{'IN':>10}{'OUT':>9}{'COST $':>11}{'% TOK':>7}"
        )
        lines.append("-" * 78)

        for stage, row in sorted(
            stages.items(),
            key=lambda kv: kv[1]["input_tokens"] + kv[1]["output_tokens"],
            reverse=True,
        ):
            stage_tokens = row["input_tokens"] + row["output_tokens"]
            pct = (stage_tokens / total_tokens * 100) if total_tokens else 0.0
            cost = f"{row['cost_usd']:.5f}" if row["cost_known"] else "  n/a"
            flag = "" if row["errors"] == 0 else f"  ({row['errors']} err)"
            lines.append(
                f"{stage[:32]:<32}{row['calls']:>6}{row['input_tokens']:>10,}"
                f"{row['output_tokens']:>9,}{cost:>11}{pct:>6.1f}%{flag}"
            )

        lines.append("-" * 78)
        total_cost = self.total_cost
        cost_str = f"${total_cost:.5f}" if total_cost is not None else "n/a"
        lines.append(
            f"{'TOTAL':<32}{len(self.calls):>6}{self.total_input:>10,}"
            f"{self.total_output:>9,}{cost_str:>11}"
        )
        if self.total_cached:
            lines.append(f"  cached input tokens: {self.total_cached:,}")
        lines.append(f"  wall clock: {round(time.time() - self.started_at, 2)}s")

        if self.any_cost_unknown:
            missing = ", ".join(report_unpriced_models()) or "unknown"
            lines.append(
                "  [!] cost incomplete - no price configured for: " + missing
            )
            lines.append(
                "      add them to config/llm_pricing.json to get dollar figures."
            )
        if self.any_estimated:
            lines.append(
                "  [!] some token counts are ESTIMATED (provider returned no usage "
                "block). Treat those as approximate."
            )
        lines.append("=" * 78)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Ambient context (run + stage)
# --------------------------------------------------------------------------

_run_var: contextvars.ContextVar[Optional[RunAccumulator]] = contextvars.ContextVar(
    "llm_run", default=None
)
_stage_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "llm_stage", default="unattributed"
)


def current_run() -> Optional[RunAccumulator]:
    return _run_var.get()


def current_stage() -> str:
    return _stage_var.get()


@contextlib.contextmanager
def llm_run(
    label: str,
    client_id: Optional[str] = None,
    run_id: Optional[str] = None,
    print_summary: bool = True,
) -> Iterator[RunAccumulator]:
    """Wrap one logical generation run so its calls aggregate together.

    Safe in async code: `contextvars` propagate into tasks created inside the
    block, so concurrent extractor/classifier workers attribute correctly.
    """
    acc = RunAccumulator(
        run_id=run_id or uuid.uuid4().hex[:12],
        label=label,
        client_id=client_id,
    )
    token = _run_var.set(acc)
    try:
        yield acc
    finally:
        _run_var.reset(token)
        if print_summary and TELEMETRY_ENABLED:
            try:
                print(acc.format_summary())
                _write_run_summary(acc)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[TELEMETRY][WARN] summary failed: {exc}")


def instrument_stream(label: str, client_id_kwarg: Optional[str] = None):
    """Decorate an async generator so every LLM call inside it forms one run.

    Used instead of wrapping the function body so the diff stays a single
    line. If the consumer abandons the stream (e.g. the client disconnects),
    `GeneratorExit` still unwinds the context manager, so a partial run is
    reported rather than silently lost.
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            client_id = kwargs.get(client_id_kwarg) if client_id_kwarg else None
            with llm_run(label, client_id=client_id):
                async for item in fn(*args, **kwargs):
                    yield item

        return wrapper

    return decorator


@contextlib.contextmanager
def llm_stage(name: str) -> Iterator[None]:
    """Attribute every LLM call made inside this block to `name`."""
    token = _stage_var.set(name)
    try:
        yield
    finally:
        _stage_var.reset(token)


# --------------------------------------------------------------------------
# Usage extraction
# --------------------------------------------------------------------------


def _extract_usage(response: Any) -> tuple[Optional[int], Optional[int], int, str]:
    """Pull (input, output, cached, model) out of a LangChain LLMResult.

    Providers differ, so several shapes are tried. Returns `None` token counts
    when the provider gave us nothing - the caller then estimates.
    """
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens = 0
    model = ""

    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        model = llm_output.get("model_name") or llm_output.get("model") or ""
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if isinstance(usage, dict) and usage:
            input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
            output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
            details = usage.get("prompt_tokens_details") or {}
            if isinstance(details, dict):
                cached_tokens = details.get("cached_tokens", 0) or 0

    # Fallback: usage_metadata on the message (LangChain >= 0.2 standard shape)
    if input_tokens is None or output_tokens is None:
        try:
            generations = getattr(response, "generations", None) or []
            for group in generations:
                for gen in group:
                    message = getattr(gen, "message", None)
                    meta = getattr(message, "usage_metadata", None)
                    if not meta:
                        continue
                    input_tokens = meta.get("input_tokens", input_tokens)
                    output_tokens = meta.get("output_tokens", output_tokens)
                    details = meta.get("input_token_details") or {}
                    if isinstance(details, dict):
                        cached_tokens = details.get("cache_read", cached_tokens) or cached_tokens
                    if not model:
                        rmeta = getattr(message, "response_metadata", None) or {}
                        model = rmeta.get("model_name") or rmeta.get("model") or model
                    if input_tokens is not None and output_tokens is not None:
                        break
        except Exception:  # pragma: no cover - defensive
            pass

    return input_tokens, output_tokens, int(cached_tokens or 0), model


def _completion_chars(response: Any) -> int:
    total = 0
    try:
        for group in getattr(response, "generations", None) or []:
            for gen in group:
                total += len(getattr(gen, "text", "") or "")
    except Exception:  # pragma: no cover - defensive
        pass
    return total


# --------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------


def _write_jsonl(path: Path, payload: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(payload + "\n")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[TELEMETRY][WARN] could not write {path}: {exc}")


def _write_run_summary(acc: RunAccumulator) -> None:
    summary_path = TELEMETRY_LOG_PATH.with_name(
        TELEMETRY_LOG_PATH.stem + "_runs.jsonl"
    )
    _write_jsonl(summary_path, json.dumps(acc.summary_dict(), ensure_ascii=False))


def _persist_supabase(call: LLMCall) -> None:
    if not TELEMETRY_TO_SUPABASE:
        return
    try:
        from app.core.supabase import supabase_admin

        supabase_admin.table("llm_calls").insert(asdict(call)).execute()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[TELEMETRY][WARN] supabase insert failed: {exc}")


def record_call(call: LLMCall) -> None:
    """Fan a completed call out to every sink. Never raises."""
    if not TELEMETRY_ENABLED:
        return
    try:
        run = current_run()
        if run is not None:
            run.add(call)

        cost = (
            f"${call.cost_usd:.6f}" if (call.cost_known and call.cost_usd is not None) else "n/a"
        )
        est = " ~est" if call.estimated else ""
        status = "" if call.ok else " FAILED"
        print(
            f"[TELEMETRY] {call.stage:<28} {call.model[:34]:<34} "
            f"in={call.input_tokens:>7,} out={call.output_tokens:>6,}"
            f" cached={call.cached_tokens:>6,} {cost:>12} "
            f"{call.latency_ms:>6}ms{est}{status}"
        )

        _write_jsonl(TELEMETRY_LOG_PATH, call.to_json())
        _persist_supabase(call)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[TELEMETRY][WARN] record_call failed: {exc}")


# --------------------------------------------------------------------------
# LangChain callback handler
# --------------------------------------------------------------------------

try:  # keep the module importable even without langchain installed
    from langchain_core.callbacks import AsyncCallbackHandler as _AsyncCallbackHandler
except Exception:  # pragma: no cover - defensive
    try:
        from langchain.callbacks.base import (  # type: ignore
            AsyncCallbackHandler as _AsyncCallbackHandler,
        )
    except Exception:
        _AsyncCallbackHandler = object  # type: ignore


class LLMTelemetryHandler(_AsyncCallbackHandler):  # type: ignore[misc]
    """Records token usage for every async LLM call.

    Every call in this codebase goes through `ainvoke`, so an async handler
    gives complete coverage. Registering a sync handler as well would
    double-count, so deliberately do not.
    """

    def __init__(self, provider: str = "", default_model: str = "") -> None:
        self.provider = provider
        self.default_model = default_model
        self._starts: Dict[str, Dict[str, Any]] = {}
        self._starts_lock = threading.Lock()

    # -- start ------------------------------------------------------------

    def _record_start(self, run_id: Any, prompts: Sequence[str], stage: str) -> None:
        with self._starts_lock:
            self._starts[str(run_id)] = {
                "t0": time.perf_counter(),
                "stage": stage,
                "prompt_chars": sum(len(p or "") for p in prompts),
            }

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        try:
            self._record_start(kwargs.get("run_id"), prompts or [], current_stage())
        except Exception:  # pragma: no cover - defensive
            pass

    async def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: Any, **kwargs: Any
    ) -> None:
        """ChatOpenAI fires this instead of `on_llm_start`."""
        try:
            flat: List[str] = []
            for batch in messages or []:
                for msg in batch or []:
                    content = getattr(msg, "content", "")
                    flat.append(content if isinstance(content, str) else str(content))
            self._record_start(kwargs.get("run_id"), flat, current_stage())
        except Exception:  # pragma: no cover - defensive
            pass

    # -- end --------------------------------------------------------------

    def _pop_start(self, run_id: Any) -> Dict[str, Any]:
        with self._starts_lock:
            return self._starts.pop(str(run_id), {})

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            start = self._pop_start(kwargs.get("run_id"))
            latency_ms = int((time.perf_counter() - start.get("t0", time.perf_counter())) * 1000)
            prompt_chars = int(start.get("prompt_chars", 0))
            stage = start.get("stage") or current_stage()

            in_tok, out_tok, cached, model = _extract_usage(response)
            completion_chars = _completion_chars(response)

            estimated = False
            if in_tok is None or out_tok is None:
                estimated = True
                in_tok = in_tok if in_tok is not None else int(prompt_chars / _CHARS_PER_TOKEN)
                out_tok = (
                    out_tok
                    if out_tok is not None
                    else int(completion_chars / _CHARS_PER_TOKEN)
                )

            model = model or self.default_model or "unknown"
            cost, cost_known = compute_cost(model, in_tok, out_tok, cached)

            run = current_run()
            record_call(
                LLMCall(
                    run_id=run.run_id if run else "no-run",
                    call_id=str(kwargs.get("run_id") or uuid.uuid4().hex[:12]),
                    ts=datetime.now(timezone.utc).isoformat(),
                    stage=stage,
                    model=model,
                    provider=self.provider,
                    input_tokens=int(in_tok or 0),
                    output_tokens=int(out_tok or 0),
                    cached_tokens=int(cached or 0),
                    total_tokens=int((in_tok or 0) + (out_tok or 0)),
                    cost_usd=cost,
                    cost_known=cost_known,
                    estimated=estimated,
                    latency_ms=latency_ms,
                    ok=True,
                    prompt_chars=prompt_chars,
                    completion_chars=completion_chars,
                    client_id=run.client_id if run else None,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[TELEMETRY][WARN] on_llm_end failed: {exc}")

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Failed calls still burn input tokens - record them."""
        try:
            start = self._pop_start(kwargs.get("run_id"))
            latency_ms = int((time.perf_counter() - start.get("t0", time.perf_counter())) * 1000)
            prompt_chars = int(start.get("prompt_chars", 0))
            stage = start.get("stage") or current_stage()
            in_tok = int(prompt_chars / _CHARS_PER_TOKEN)
            cost, cost_known = compute_cost(self.default_model, in_tok, 0, 0)

            run = current_run()
            record_call(
                LLMCall(
                    run_id=run.run_id if run else "no-run",
                    call_id=str(kwargs.get("run_id") or uuid.uuid4().hex[:12]),
                    ts=datetime.now(timezone.utc).isoformat(),
                    stage=stage,
                    model=self.default_model or "unknown",
                    provider=self.provider,
                    input_tokens=in_tok,
                    output_tokens=0,
                    cached_tokens=0,
                    total_tokens=in_tok,
                    cost_usd=cost,
                    cost_known=cost_known,
                    estimated=True,
                    latency_ms=latency_ms,
                    ok=False,
                    error=f"{type(error).__name__}: {error}"[:500],
                    prompt_chars=prompt_chars,
                    client_id=run.client_id if run else None,
                )
            )
        except Exception:  # pragma: no cover - defensive
            pass


def telemetry_callbacks(provider: str = "", model: str = "") -> List[Any]:
    """Callback list to hand to a `ChatOpenAI` constructor.

    Returns `[]` when telemetry is disabled so the wiring is a no-op.
    """
    if not TELEMETRY_ENABLED:
        return []
    try:
        return [LLMTelemetryHandler(provider=provider, default_model=model)]
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[TELEMETRY][WARN] could not build handler: {exc}")
        return []


# --------------------------------------------------------------------------
# Offline report:  python -m app.core.llm_telemetry [logs/llm_calls.jsonl]
# --------------------------------------------------------------------------


def _report(path: Path) -> int:
    if not path.exists():
        print(f"No telemetry log at {path}. Run a generation first.")
        return 1

    calls: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            calls.append(json.loads(line))
        except Exception:
            continue

    if not calls:
        print(f"{path} contains no parsable records.")
        return 1

    def agg(key: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for c in calls:
            row = out.setdefault(
                str(c.get(key)),
                {"calls": 0, "in": 0, "out": 0, "cost": 0.0, "known": True, "err": 0},
            )
            row["calls"] += 1
            row["in"] += c.get("input_tokens", 0)
            row["out"] += c.get("output_tokens", 0)
            if not c.get("ok", True):
                row["err"] += 1
            if c.get("cost_known") and c.get("cost_usd") is not None:
                row["cost"] += c["cost_usd"]
            else:
                row["known"] = False
        return out

    runs = {c.get("run_id") for c in calls}
    total_in = sum(c.get("input_tokens", 0) for c in calls)
    total_out = sum(c.get("output_tokens", 0) for c in calls)

    print("=" * 78)
    print(f"LLM telemetry report - {path}")
    print(f"{len(calls):,} calls across {len(runs)} run(s)")
    print("=" * 78)

    for title, key in (("BY STAGE", "stage"), ("BY MODEL", "model")):
        print(f"\n{title}")
        print(
            f"{'':<34}{'CALLS':>7}{'IN':>12}{'OUT':>10}{'COST $':>12}{'% TOK':>8}"
        )
        print("-" * 78)
        rows = agg(key)
        for name, row in sorted(
            rows.items(), key=lambda kv: kv[1]["in"] + kv[1]["out"], reverse=True
        ):
            tok = row["in"] + row["out"]
            pct = (tok / (total_in + total_out) * 100) if (total_in + total_out) else 0
            cost = f"{row['cost']:.5f}" if row["known"] else "n/a"
            err = "" if not row["err"] else f"  ({row['err']} err)"
            print(
                f"{name[:34]:<34}{row['calls']:>7}{row['in']:>12,}"
                f"{row['out']:>10,}{cost:>12}{pct:>7.1f}%{err}"
            )

    print("-" * 78)
    known_cost = sum(
        c["cost_usd"] for c in calls if c.get("cost_known") and c.get("cost_usd")
    )
    complete = all(c.get("cost_known") for c in calls)
    print(f"TOTAL tokens: in={total_in:,}  out={total_out:,}")
    print(f"TOTAL cost:   ${known_cost:.5f}" + ("" if complete else "  (INCOMPLETE)"))

    if len(runs) > 1:
        print(f"Per run avg:  ${known_cost / len(runs):.5f}")
    if not complete:
        unpriced = sorted(
            {c.get("model", "?") for c in calls if not c.get("cost_known")}
        )
        print("\n[!] No price configured for: " + ", ".join(unpriced))
        print("    Add them to config/llm_pricing.json for dollar figures.")
    if any(c.get("estimated") for c in calls):
        n = sum(1 for c in calls if c.get("estimated"))
        print(f"\n[!] {n} call(s) had ESTIMATED tokens (no usage block returned).")
    print("=" * 78)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TELEMETRY_LOG_PATH
    raise SystemExit(_report(target))
