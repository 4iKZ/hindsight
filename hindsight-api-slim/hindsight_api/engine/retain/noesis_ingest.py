"""Noesis Stage 1 event ingestion (requirement 02, 04A latest contract).

Replaces the old hyper_extract directed-graph side path on the Hindsight
production retain main chain. For every non-empty content item this module:

1. builds a ``NoesisInputItem`` envelope (content + observed_at + identity);
2. awaits the authoritative ``hyperextract.noesis`` extraction off the event
   loop (``asyncio.to_thread`` — no daemon threads, no fire-and-forget);
3. routes the outcome: ``[]`` is a silent success, hyper-extract alerts and
   hypotheses are recorded in ``noesis_core.ingestion_alerts``, and every fact
   component is written inside one short transaction to ``noesis_core.events``
   / ``atoms`` / ``event_atoms`` — events are sequence-numbered via
   ``INSERT ... RETURNING event_id`` (04A: no ingestion_key, no replay,
   duplicate input is a new event by contract);
4. isolates every failure from the native Hindsight retain: failures become
   alerts (never exceptions) and the retain pipeline continues untouched.

The Noesis database is a dedicated asyncpg pool against ``noesis`` /
``noesis_core``; LLM calls and deterministic validation always run outside any
database transaction. Application startup never executes DDL.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

import asyncpg

from ...utils import mask_network_location
from .noesis_identity_vector import (
    IdentityConfigInvalid,
    IdentityEmbedError,
    IdentityServiceUnavailable,
    NoesisIdentityClient,
    is_transport_kind,
)

try:  # hyperextract is an optional dependency; the public API is imported, never copied
    from hyperextract.noesis import extract_noesis_components
except ImportError:  # pragma: no cover - surfaced as a noesis_llm_config_invalid alert
    extract_noesis_components = None

logger = logging.getLogger(__name__)

CONTRACT_VERSION = "noesis-event-closure-v1"
DEFAULT_NOESIS_SCHEMA = "noesis_core"
DEFAULT_NOESIS_TIMEZONE = "Asia/Shanghai"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Providers whose native clients cannot be built by Hyper-Extract's
# create_llm (openai-compatible and anthropic providers can). Anything here
# yields a noesis_llm_config_invalid alert instead of a silently swapped model.
_UNSAFE_PROVIDERS = frozenset({"gemini", "vertexai", "bedrock"})


class NoesisConfigError(Exception):
    """The retain LLM / Noesis deployment configuration cannot run extraction."""


class SchemaPreflightError(Exception):
    """A required object is missing from the target schema (R02-06).

    Raised only when the probe can name the exact missing object, so the
    process is flagged as not-ready instead of silently skipping ingestion.
    """


@dataclass(frozen=True)
class NoesisInputItem:
    """Per-item ingestion envelope (requirement 02 §6)."""

    bank_id: str
    content: str
    observed_at: datetime
    operation_id: str | None
    document_id: str | None
    item_index: int
    source: str = "hindsight_retain"


@dataclass(frozen=True)
class TimeResolution:
    """Result of the frozen event_time algorithm (requirement 02 §9.2)."""

    event_time: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure helpers: canonical JSON, keys, envelopes
# ---------------------------------------------------------------------------

def validate_schema_identifier(schema: str) -> bool:
    """Only plain SQL identifiers are accepted for the application schema."""
    return bool(isinstance(schema, str) and _IDENTIFIER_RE.fullmatch(schema))


def canonical_json_bytes(payload: Any) -> bytes:
    """UTF-8, ensure_ascii=False, sorted keys, no insignificant whitespace."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def stamp_noesis_queue_metadata(contents: list[dict[str, Any]], *, captured_at: datetime | None = None) -> None:
    """Freeze retry-sensitive Noesis identity fields in a queued task payload."""
    observed_at = _iso_utc(captured_at or _utc_now())
    for index, item in enumerate(contents):
        item.setdefault("_noesis_item_index", index)
        if item.get("event_date") is None:
            item.setdefault("_noesis_observed_at", observed_at)


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_alert_dedupe_key(
    *, item: NoesisInputItem, stage: str, alert_code: str, component_index: int | None
) -> str:
    """One open alert per retried input problem — requirement 02 §13.1."""
    payload = {
        "bank_id": item.bank_id,
        "operation_id": item.operation_id,
        "document_id": item.document_id,
        "item_index": item.item_index,
        "observed_at": _iso_utc(item.observed_at),
        "content_sha256": _content_sha256(item.content),
        "stage": stage,
        "alert_code": alert_code,
        "component_index": component_index if component_index is not None else -1,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_source_envelope(*, item: NoesisInputItem, attempts: int) -> dict[str, Any]:
    """Safe alert envelope: identity + hashes only, never content or secrets."""
    return {
        "bank_id": item.bank_id,
        "operation_id": item.operation_id,
        "document_id": item.document_id,
        "item_index": item.item_index,
        "content_sha256": _content_sha256(item.content),
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# event_time resolution (requirement 02 §9.2)
# ---------------------------------------------------------------------------

def resolve_event_time(*, observed_at: datetime, atoms: Any, analyzer: Any, timezone_name: str) -> TimeResolution:
    """Frozen algorithm: observed_at + modifier atoms, analyzer injected, no LLM."""
    try:
        business_tz = ZoneInfo(timezone_name)
        reference_naive = observed_at.astimezone(business_tz).replace(tzinfo=None)
    except Exception:
        return _time_fallback(observed_at, [], "event_time_parse_failed", "configured noesis timezone is unusable")

    modifiers = [atom.text for atom in atoms if getattr(atom, "role", None) == "modifier"]
    constraints: list[tuple[datetime, datetime | None, str]] = []
    for text in modifiers:
        try:
            analysis = analyzer.analyze(text, reference_date=reference_naive)
        except Exception as error:
            return _time_fallback(
                observed_at, modifiers, "event_time_parse_failed", f"time analyzer failed: {type(error).__name__}"
            )
        constraint = getattr(analysis, "temporal_constraint", None)
        if constraint is None and getattr(analysis, "start_date", None) is not None:
            constraint = analysis  # analyzers may hand back the constraint object itself
        if constraint is None:
            continue  # not a temporal modifier — leaves event_time untouched
        constraints.append(
            (
                _as_utc(constraint.start_date, business_tz),
                _as_utc(constraint.end_date, business_tz) if constraint.end_date is not None else None,
                text,
            )
        )

    if not constraints:
        return TimeResolution(
            event_time=observed_at,
            metadata={
                "strategy": "observed_at",
                "matched_atoms": [],
                "start": None,
                "end": None,
                "fallback_reason": None,
            },
            warnings=[],
        )

    # Multiple atoms resolving to the same start are one constraint (§9.2 #3).
    deduped: dict[datetime, tuple[datetime | None, list[str]]] = {}
    for start, end, text in constraints:
        entry = deduped.setdefault(start, (end, []))
        entry[1].append(text)

    if len(deduped) > 1:
        return _time_fallback(
            observed_at,
            [text for _, (_, texts) in deduped.items() for text in texts],
            "event_time_conflict",
            "conflicting time constraints resolved from modifier atoms",
        )

    start, (end, matched) = next(iter(deduped.items()))
    if start > observed_at + timedelta(hours=24):
        return _time_fallback(
            observed_at, matched, "event_time_out_of_range", "resolved event time is more than 24h in the future"
        )
    return TimeResolution(
        event_time=start,
        metadata={
            "strategy": "modifier_atom",
            "matched_atoms": matched,
            "start": _iso_utc(start),
            "end": _iso_utc(end),
            "fallback_reason": None,
        },
        warnings=[],
    )


def _time_fallback(observed_at: datetime, matched_atoms: list[str], alert_code: str, reason: str) -> TimeResolution:
    return TimeResolution(
        event_time=observed_at,
        metadata={
            "strategy": "fallback",
            "matched_atoms": matched_atoms,
            "start": None,
            "end": None,
            "fallback_reason": reason,
        },
        warnings=[
            {
                "alert_code": alert_code,
                "message": reason,
                "details": {"matched_atoms": matched_atoms, "reason": reason},
            }
        ],
    )


def _as_utc(value: datetime, business_tz: ZoneInfo) -> datetime:
    """Naive parse results are business-timezone wall times; aware ones convert."""
    if value.tzinfo is None:
        return value.replace(tzinfo=business_tz).astimezone(UTC)
    return value.astimezone(UTC)


# ---------------------------------------------------------------------------
# Batch normalization (requirement 02 §6)
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_observed_at(value: Any) -> datetime:
    """Mirror orchestrator.parse_datetime_flexible for raw dicts (no import cycle)."""
    if isinstance(value, datetime):
        return (value.replace(tzinfo=UTC) if value.tzinfo is None else value).astimezone(UTC)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed).astimezone(UTC)
    raise TypeError(f"Expected datetime or string, got {type(value).__name__}")


async def normalize_items(
    contents_dicts: Any,
    *,
    bank_id: str,
    operation_id: str | None,
    batch_document_id: str | None,
    clock: Callable[[], datetime] | None = None,
) -> list[NoesisInputItem]:
    """One NoesisInputItem per non-empty dict; blank items are silent skips."""
    items: list[NoesisInputItem] = []
    for index, entry in enumerate(contents_dicts or []):
        content = entry.get("content") if isinstance(entry, dict) else None
        if not isinstance(content, str) or not content.strip():
            continue
        observed_at = await _resolve_observed_at(
            operation_id=operation_id,
            document_id=entry.get("document_id") or batch_document_id,
            content=content,
            explicit=entry.get("event_date", entry.get("_noesis_observed_at")),
            clock=clock,
        )
        items.append(
            NoesisInputItem(
                bank_id=bank_id,
                content=content,
                observed_at=observed_at,
                operation_id=operation_id,
                document_id=entry.get("document_id") or batch_document_id,
                item_index=(
                    entry["_noesis_item_index"]
                    if isinstance(entry.get("_noesis_item_index"), int) and entry["_noesis_item_index"] >= 0
                    else index
                ),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Extraction client (requirement 02 §7)
# ---------------------------------------------------------------------------

def _resolve_llm_spec(llm_config: Any) -> tuple[str, str, str, str]:
    config = llm_config
    members = getattr(llm_config, "members", None)
    if members:
        config = members[0]  # multi-LLM chains: extraction uses the primary member
    return (
        getattr(config, "provider", "") or "",
        getattr(config, "model", "") or "",
        getattr(config, "base_url", "") or "",
        getattr(config, "api_key", "") or "",
    )


def _production_extract_once_factory(llm_config: Any) -> Callable[[str], object]:
    """Reuse the resolved retain LLM through Hyper-Extract's public API only."""
    provider, model, base_url, api_key = _resolve_llm_spec(llm_config)
    if not provider or not model:
        raise NoesisConfigError("noesis extraction requires a resolved retain LLM provider and model")
    if provider.lower() in _UNSAFE_PROVIDERS:
        raise NoesisConfigError(f"retain LLM provider '{provider}' cannot be created by the Hyper-Extract client")
    try:
        from hyperextract import create_llm as he_create_llm
        from hyperextract.noesis import create_noesis_extractor
    except ImportError as error:
        raise NoesisConfigError(f"hyperextract.noesis is unavailable: {error}") from error
    client = he_create_llm(
        {"provider": provider, "model": model, "base_url": base_url},
        api_key=api_key,
        temperature=0,
    )
    return create_noesis_extractor(llm_client=client)


# ---------------------------------------------------------------------------
# asyncpg pool lifecycle (requirement 02 §14.3)
# ---------------------------------------------------------------------------

_pool: Any | None = None
_pool_lock = asyncio.Lock()

async def _resolve_observed_at(
    *,
    operation_id: str | None,
    document_id: str | None,
    content: str,
    explicit: Any,
    clock: Callable[[], datetime] | None = None,
) -> datetime:
    """Return the item's observed_at.

    An explicit ``event_date`` (or queue-persisted internal timestamp) wins.
    Otherwise the caller's batch-boundary clock is used once for this call.
    """
    if explicit is not None:
        return _parse_observed_at(explicit)
    return (clock or _utc_now)()



async def _open_pool(config: Any) -> Any:
    return await asyncpg.create_pool(
        dsn=config.noesis_database_url,
        min_size=config.noesis_pool_min_size,
        max_size=config.noesis_pool_max_size,
        command_timeout=config.noesis_command_timeout,
    )


async def _get_pool(config: Any) -> Any:
    global _pool
    async with _pool_lock:
        if _pool is None:
            _pool = await _open_pool(config)
        elif getattr(_pool, "is_closed", lambda: False)():
            _pool = await _open_pool(config)
        return _pool


async def close_noesis_pool() -> None:
    """Idempotent shutdown, called from MemoryEngine.close().

    Closes the dedicated asyncpg pool AND the shared identity-vector client
    (requirement 03 §5.3: one lazy client per process, retired together).
    """
    global _pool, _identity_client, _identity_client_key
    async with _pool_lock:
        if _pool is not None and not getattr(_pool, "is_closed", lambda: False)():
            try:
                await _pool.close()
            except Exception as error:
                logger.error("noesis pool close failed: %s", type(error).__name__)
        _pool = None
        _preflight_cache.clear()
    async with _identity_client_lock:
        if _identity_client is not None:
            try:
                await _identity_client.aclose()
            except Exception as error:
                logger.error("noesis identity client close failed: %s", type(error).__name__)
        _identity_client = None
        _identity_client_key = None


async def _acquire_pool(config: Any, pool_factory: Any) -> Any:
    if pool_factory is not None:
        return await pool_factory(config)
    return await _get_pool(config)


# ---------------------------------------------------------------------------
# Identity-vector client lifecycle (requirement 03 §5.3/§10)
# ---------------------------------------------------------------------------

_IDENTITY_KIND = "identity"


@dataclass(frozen=True)
class IdentitySpec:
    """The configured identity-vector generation (model/revision/dimension)."""

    base_url: str
    model: str
    revision: str
    dimension: int


def _identity_spec_from_config(config: Any) -> IdentitySpec:
    from ... import config as hindsight_config

    return IdentitySpec(
        base_url=(
            getattr(config, "noesis_embedding_base_url", None)
            or hindsight_config.DEFAULT_NOESIS_EMBEDDING_BASE_URL
        ),
        model=(
            getattr(config, "noesis_embedding_model", None)
            or hindsight_config.DEFAULT_NOESIS_EMBEDDING_MODEL
        ),
        revision=(
            getattr(config, "noesis_embedding_revision", None)
            or hindsight_config.DEFAULT_NOESIS_EMBEDDING_REVISION
        ),
        dimension=int(
            getattr(config, "noesis_embedding_dimension", None)
            or hindsight_config.DEFAULT_NOESIS_EMBEDDING_DIMENSION
        ),
    )


_identity_client: Any | None = None
_identity_client_key: tuple | None = None
_identity_client_lock = asyncio.Lock()


def _identity_client_key_from_config(config: Any, spec: IdentitySpec) -> tuple:
    return (
        spec.base_url,
        spec.model,
        spec.revision,
        spec.dimension,
        float(getattr(config, "noesis_embedding_timeout_seconds", 2.0) or 2.0),
        int(getattr(config, "noesis_embedding_max_retries", 1) or 0),
        getattr(config, "noesis_embedding_api_key", "") or "",
    )


async def _get_identity_client(config: Any, spec: IdentitySpec) -> Any:
    """One shared lazily-built client per process; a config change builds a
    fresh client (whose /health cache is naturally cold) and retires the old."""
    global _identity_client, _identity_client_key
    key = _identity_client_key_from_config(config, spec)
    async with _identity_client_lock:
        if _identity_client is None or _identity_client_key != key:
            stale = _identity_client
            _identity_client = NoesisIdentityClient(
                base_url=spec.base_url,
                model=spec.model,
                revision=spec.revision,
                dimension=spec.dimension,
                timeout_seconds=float(getattr(config, "noesis_embedding_timeout_seconds", 2.0) or 2.0),
                max_retries=int(getattr(config, "noesis_embedding_max_retries", 1) or 0),
                api_key=getattr(config, "noesis_embedding_api_key", "") or "",
            )
            _identity_client_key = key
            if stale is not None:
                try:
                    await stale.aclose()
                except Exception as error:
                    logger.error("stale noesis identity client close failed: %s", type(error).__name__)
        return _identity_client


async def _acquire_identity_client(config: Any, spec: IdentitySpec, factory: Any) -> Any:
    if factory is not None:
        return await factory(config)
    return await _get_identity_client(config, spec)


# ---------------------------------------------------------------------------
# Identity profile gate + pre-transaction vector preparation (requirement 03
# §8.1/§10.3). Everything here is read-only or a tiny independent claim
# transaction; bge HTTP happens outside any business transaction.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ProfileGate:
    allowed: bool
    db_profile: dict[str, Any] | None


async def _identity_profile_gate(pool: Any, schema: str, spec: IdentitySpec) -> _ProfileGate:
    """Config, live service, and DB profile must agree before vectors flow.

    Missing profile + zero existing E/P vectors → safe claim (§10.3.1).
    Missing profile + existing vectors → refuse to guess their source (§10.3.2).
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(_sql(schema, _PROFILE_SELECT), _IDENTITY_KIND)
        if row is None:
            non_null = await conn.fetchval(_sql(schema, _EP_NON_NULL_COUNT))
            if non_null:
                return _ProfileGate(allowed=False, db_profile=None)
            async with conn.transaction():
                await conn.fetchrow(_sql(schema, _PROFILE_CLAIM), spec.model, spec.revision, spec.dimension)
            row = await conn.fetchrow(_sql(schema, _PROFILE_SELECT), _IDENTITY_KIND)
            if row is None:  # pragma: no cover - claim is atomic with its conflict rule
                raise RuntimeError("identity profile claim did not produce a row")
        profile = {
            "model_name": row["model_name"],
            "model_revision": row["model_revision"],
            "dimension": int(row["dimension"]),
            "status": row["status"],
        }
    matches = (profile["model_name"], profile["model_revision"], profile["dimension"]) == (
        spec.model,
        spec.revision,
        spec.dimension,
    )
    return _ProfileGate(allowed=matches and profile["status"] == "ready", db_profile=profile)


def _identity_alert(
    *,
    stage: str,
    alert_code: str,
    severity: str,
    message: str,
    spec: IdentitySpec,
    attempted: int = 0,
    succeeded: int = 0,
    failed_atoms: list[dict[str, Any]] | None = None,
    db_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate alert payload; never carries credentials, bodies, or stacks."""
    details: dict[str, Any] = {
        "attempted": attempted,
        "succeeded": succeeded,
        "failed_atoms": list(failed_atoms or []),
        "model": spec.model,
        "revision": spec.revision,
        "base_url": mask_network_location(spec.base_url),
    }
    if db_profile is not None:
        details["db_profile"] = db_profile
        details["config_profile"] = {"model_name": spec.model, "model_revision": spec.revision, "dimension": spec.dimension}
    return {"stage": stage, "alert_code": alert_code, "severity": severity, "message": message, "details": details}


@dataclass(frozen=True)
class _VectorPrep:
    vectors: dict[tuple[str, str], list[float]]
    alert: dict[str, Any] | None


async def _prepare_identity_vectors(
    *,
    pool: Any,
    schema: str,
    client: Any,
    spec: IdentitySpec,
    literals: list[tuple[str, str]],
) -> _VectorPrep:
    """Health gate → profile gate → atom-state precheck → sequential bge calls.

    All HTTP and probes happen before (and outside) the fact transaction.
    """
    if client is None:
        return _VectorPrep(
            {},
            _identity_alert(
                stage="identity_vector",
                alert_code="identity_vector_unavailable",
                severity="warning",
                message="identity vector client could not be constructed; literals register with NULL vectors",
                spec=spec,
            ),
        )
    try:
        await client.ensure_ready()
    except IdentityConfigInvalid:
        return _VectorPrep(
            {},
            _identity_alert(
                stage="noesis_config",
                alert_code="identity_vector_config_invalid",
                severity="error",
                message="identity service /health contradicts the configured generation; vectors disabled for this process",
                spec=spec,
            ),
        )
    except IdentityServiceUnavailable:
        return _VectorPrep(
            {},
            _identity_alert(
                stage="identity_vector",
                alert_code="identity_vector_unavailable",
                severity="warning",
                message="identity service /health unreachable; literals register with NULL vectors",
                spec=spec,
            ),
        )

    gate = await _identity_profile_gate(pool, schema, spec)
    if not gate.allowed:
        return _VectorPrep(
            {},
            _identity_alert(
                stage="identity_vector",
                alert_code="identity_vector_profile_mismatch",
                severity="error",
                message="database identity profile does not match the running config or is rebuilding; vector writes suspended",
                spec=spec,
                db_profile=gate.db_profile,
            ),
        )

    vectors: dict[tuple[str, str], list[float]] = {}
    failed: list[dict[str, Any]] = []
    if literals:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                _sql(schema, _ATOM_STATE_SELECT),
                [text for text, _ in literals],
                [atom_type for _, atom_type in literals],
            )
        state = {(row["text"], row["atom_type"]): row for row in rows}
        need = [literal for literal in literals if not state[literal]["has_embedding"]]
        for text, atom_type in need:  # sequential by design (§5.3): no GPU fan-out
            if not text.strip():
                failed.append({"text": text, "atom_type": atom_type, "error_kind": "empty_literal"})
                continue
            try:
                vectors[(text, atom_type)] = await client.embed(text, atom_type)
            except IdentityEmbedError as error:
                failed.append({"text": text, "atom_type": atom_type, "error_kind": error.error_kind})
    if not failed:
        return _VectorPrep(vectors, None)

    attempted = len(vectors) + len(failed)
    if not vectors and all(is_transport_kind(entry["error_kind"]) for entry in failed):
        return _VectorPrep(
            vectors,
            _identity_alert(
                stage="identity_vector",
                alert_code="identity_vector_unavailable",
                severity="warning",
                message="identity service unreachable for the whole batch; literals register with NULL vectors",
                spec=spec,
                attempted=attempted,
                succeeded=len(vectors),
                failed_atoms=failed,
            ),
        )
    return _VectorPrep(
        vectors,
        _identity_alert(
            stage="identity_vector",
            alert_code="identity_vector_failed",
            severity="warning",
            message="some literals could not be embedded; they register with NULL vectors and self-heal on recurrence",
            spec=spec,
            attempted=attempted,
            succeeded=len(vectors),
            failed_atoms=failed,
        ),
    )


def _vector_literal(vector: Any) -> str:
    """asyncpg speaks to pgvector through an explicit ``$3::vector`` cast of a
    text literal; ``repr(float(...))`` keeps full precision and valid syntax."""
    return "[" + ",".join(repr(float(component)) for component in vector) + "]"


# ---------------------------------------------------------------------------
# Read-only schema preflight (requirement 02 §12 / R02-06)
# ---------------------------------------------------------------------------

_preflight_lock = asyncio.Lock()
_preflight_cache: dict[tuple[int, str], bool] = {}

# The object identity the application depends on for the four ingest tables
# plus the requirement 03 identity profile gate.
_PREFLIGHT_TABLES = ("atoms", "events", "event_atoms", "ingestion_alerts", "embedding_profiles")
_PREFLIGHT_EXTENSIONS = ("vector", "roaringbitmap", "timescaledb", "pg_ripple")
_PREFLIGHT_COLUMNS = {
    "atoms": ("text", "atom_type", "embedding"),
    "events": ("event_time", "data"),
    "event_atoms": ("event_id", "occurrence_id", "atom_id", "role_type", "target_occ"),
    "ingestion_alerts": ("dedupe_key", "stage", "alert_code", "severity", "message", "details"),
    "embedding_profiles": ("embedding_kind", "model_name", "model_revision", "dimension", "status", "updated_at"),
}
# 04A: JSON role -> event_atoms.role_type "char" enum (requirement 04A §6.4).
_ROLE_TYPE = {"agent": "A", "patient": "P", "predicate": "R", "modifier": "M"}


async def _run_schema_preflight(pool: Any, schema: str, expected_dimension: int = 1024) -> None:
    """Verify the target schema is usable by this ingestion module.

    Read-only (SELECT on information_schema / pg_catalog). Raises
    SchemaPreflightError naming the first missing object; never ALTERs, never
    prints a DSN or password. ``expected_dimension`` pins the declared width
    of atoms.embedding (requirement 03 §11), verified against the live
    catalog as ``vector(N)`` (format confirmed against the remote database).
    """
    async with pool.acquire() as conn:
        for extension in _PREFLIGHT_EXTENSIONS:
            installed = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = $1)", extension
            )
            if not installed:
                raise SchemaPreflightError(f"required extension '{extension}' is not installed")

        schema_exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = $1)", schema
        )
        if not schema_exists:
            raise SchemaPreflightError(f"schema '{schema}' does not exist")

        for table in _PREFLIGHT_TABLES:
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = $1 AND table_name = $2)",
                schema,
                table,
            )
            if not exists:
                raise SchemaPreflightError(f"table '{schema}.{table}' does not exist")
            for column in _PREFLIGHT_COLUMNS[table]:
                col = await conn.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = $1 AND table_name = $2 AND column_name = $3)",
                    schema,
                    table,
                    column,
                )
                if not col:
                    raise SchemaPreflightError(f"column '{schema}.{table}.{column}' does not exist")

        # 04A: events is a sequence-numbered hypertable without a unique key;
        # the plain event_id index is the object the writer depends on.
        event_id_index = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_indexes "
            "WHERE schemaname = $1 AND tablename = 'events' AND indexname = 'idx_events_event_id')",
            schema,
        )
        if not event_id_index:
            raise SchemaPreflightError(f"index 'idx_events_event_id' missing on '{schema}.events'")

        dedupe_not_null = await conn.fetchval(
            "SELECT (is_nullable = 'NO') FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = 'ingestion_alerts' AND column_name = 'dedupe_key'",
            schema,
        )
        dedupe_unique = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_constraint "
            "WHERE conname = 'ingestion_alerts_dedupe_key_unique' "
            "AND conrelid = to_regclass($1))",
            f"{schema}.ingestion_alerts",
        )
        if dedupe_not_null is not True or not dedupe_unique:
            raise SchemaPreflightError(
                f"ingestion_alerts.dedupe_key is not NOT NULL+UNIQUE on '{schema}.ingestion_alerts'"
            )

        atom_pair_unique = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_constraint "
            "WHERE conname = 'atoms_text_type_unique' AND conrelid = to_regclass($1))",
            f"{schema}.atoms",
        )
        if not atom_pair_unique:
            raise SchemaPreflightError(f"unique constraint 'atoms_text_type_unique' missing on '{schema}.atoms'")

        hypertable = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM timescaledb_information.hypertables "
            "WHERE hypertable_schema = $1 AND hypertable_name = 'events')",
            schema,
        )
        if not hypertable:
            raise SchemaPreflightError(f"'{schema}.events' is not a TimescaleDB hypertable")

        retention = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM timescaledb_information.jobs "
            "WHERE hypertable_schema = $1 AND hypertable_name = 'events' "
            "AND proc_name = 'policy_retention' AND config->>'drop_after' = '90 days')",
            schema,
        )
        if not retention:
            raise SchemaPreflightError(f"90-day retention policy missing on '{schema}.events'")

        # Requirement 03 §11: the declared embedding width must match the
        # configured generation, so a wrong-dimension schema can never be
        # silently written.
        declared = await conn.fetchval(
            "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
            "WHERE a.attrelid = to_regclass($1) AND a.attname = 'embedding'",
            f"{schema}.atoms",
        )
        if declared != f"vector({expected_dimension})":
            raise SchemaPreflightError(
                f"'{schema}.atoms.embedding' declares {declared!r}, expected 'vector({expected_dimension})'"
            )


async def _ensure_schema_ready(pool: Any, schema: str, expected_dimension: int = 1024) -> bool:
    """Probe once per pool/schema. Unknown failures skip Noesis for this call
    and are deliberately not cached, so the next batch retries the probe."""
    cache_key = (id(pool), schema, expected_dimension)
    async with _preflight_lock:
        if cache_key in _preflight_cache:
            return _preflight_cache[cache_key]
        try:
            await _run_schema_preflight(pool, schema, expected_dimension=expected_dimension)
        except SchemaPreflightError as error:
            logger.error("noesis schema preflight failed (ingestion disabled): %s", error)
            _preflight_cache[cache_key] = False
            return False
        except Exception as error:
            logger.warning("noesis schema preflight could not be completed: %s", type(error).__name__)
            return False
        _preflight_cache[cache_key] = True
        return True


# ---------------------------------------------------------------------------
# Store layer: the only SQL this module runs
# ---------------------------------------------------------------------------

def _sql(schema: str, statement: str) -> str:
    return statement.format(s=schema)


# 04A: events are sequence-numbered (source 'L' = LLM extraction, category
# 'R' = raw — the frozen "char" enum values of the latest DDL). No unique key,
# no replay: a duplicate input lands a new event by contract. The
# ``::text::"char"`` casts bind Python str parameters to the internal
# one-byte "char" columns (asyncpg only accepts bytes for a direct bind).
_EVENT_INSERT = (
    "INSERT INTO {s}.events (event_time, data, source, category) "
    "VALUES ($1, $2::jsonb, $3::text::\"char\", $4::text::\"char\") "
    "RETURNING event_id"
)
# Requirement 03 §8.3 + 04A: new rows carry the freshly computed vector (or
# NULL on bge failure); an existing non-NULL embedding is never overwritten; a
# NULL embedding is backfilled (self-heal) when a vector arrives. Atoms carry
# no support_count (support is derived from the Cooccurrence Bitmap later).
_ATOM_UPSERT = (
    "INSERT INTO {s}.atoms (text, atom_type, embedding, status) "
    "VALUES ($1, $2::text::\"char\", $3::vector, 'A') "
    "ON CONFLICT (text, atom_type) DO UPDATE SET "
    "embedding = CASE "
    "WHEN {s}.atoms.embedding IS NULL AND EXCLUDED.embedding IS NOT NULL THEN EXCLUDED.embedding "
    "ELSE {s}.atoms.embedding END "
    "RETURNING atom_id"
)
# Requirement 03 §8.2: pre-transaction read-only atom state probe.
_ATOM_STATE_SELECT = (
    "SELECT t.text, t.atom_type, "
    "(a.atom_id IS NOT NULL) AS atom_exists, (a.embedding IS NOT NULL) AS has_embedding "
    "FROM unnest($1::text[], $2::text[]) AS t(text, atom_type) "
    "LEFT JOIN {s}.atoms a ON a.text = t.text AND a.atom_type = t.atom_type"
)
# Requirement 03 §10.3: the single-row identity model-generation gate.
_PROFILE_SELECT = (
    "SELECT model_name, model_revision, dimension, status FROM {s}.embedding_profiles "
    "WHERE embedding_kind = $1"
)
_PROFILE_WRITE_GUARD = (
    "SELECT model_name, model_revision, dimension, status FROM {s}.embedding_profiles "
    "WHERE embedding_kind = $1 FOR SHARE"
)
_PROFILE_CLAIM = (
    "INSERT INTO {s}.embedding_profiles (embedding_kind, model_name, model_revision, dimension, status) "
    "VALUES ('identity', $1, $2, $3, 'ready') "
    "ON CONFLICT (embedding_kind) DO NOTHING RETURNING embedding_kind"
)
_EP_NON_NULL_COUNT = "SELECT count(*) FROM {s}.atoms WHERE embedding IS NOT NULL AND atom_type IN ('E', 'P')"
# 04A: target_occ is the atoms[].target_occ value verbatim; role_type carries
# the mapped "char" value (see _ROLE_TYPE) bound via the text cast.
_EVENT_ATOM_INSERT = (
    "INSERT INTO {s}.event_atoms (event_id, occurrence_id, atom_id, role_type, target_occ) "
    "VALUES ($1, $2, $3, $4::text::\"char\", $5)"
)
_ALERT_INSERT = (
    "INSERT INTO {s}.ingestion_alerts (dedupe_key, event_id, stage, alert_code, severity, message, details, status) "
    "VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, 'open') "
    "ON CONFLICT (dedupe_key) DO NOTHING RETURNING alert_id"
)


def _build_event_data(
    *, item: NoesisInputItem, component_index: int, component_json: dict, time_metadata: dict
) -> dict:
    """Canonical envelope — requirement 02 §10.4."""
    return {
        "contract_version": CONTRACT_VERSION,
        "bank_id": item.bank_id,
        "operation_id": item.operation_id,
        "document_id": item.document_id,
        "item_index": item.item_index,
        "component_index": component_index,
        "source_text": item.content,
        "observed_at": _iso_utc(item.observed_at),
        "time_resolution": time_metadata,
        "component": component_json,
    }


async def _write_alert(
    conn: Any,
    schema: str,
    *,
    dedupe_key: str,
    event_id: int | None,
    stage: str,
    alert_code: str,
    severity: str,
    message: str,
    details: dict,
) -> None:
    await conn.fetchrow(
        _sql(schema, _ALERT_INSERT),
        dedupe_key,
        event_id,
        stage,
        alert_code,
        severity,
        message,
        json.dumps(details, ensure_ascii=False),
    )


async def _write_alert_safe(pool: Any, schema: str, item: NoesisInputItem, **kwargs: Any) -> None:
    """Independent-connection alert write; failure is logged, never raised."""
    dedupe_key = compute_alert_dedupe_key(
        item=item,
        stage=kwargs["stage"],
        alert_code=kwargs["alert_code"],
        component_index=kwargs.pop("component_index", None),
    )
    details = kwargs.pop("details")
    try:
        async with pool.acquire() as conn:
            await _write_alert(conn, schema, dedupe_key=dedupe_key, details=details, **kwargs)
    except Exception as error:
        logger.error(
            "noesis alert write failed (%s/%s): %s",
            kwargs.get("stage"),
            kwargs.get("alert_code"),
            type(error).__name__,
        )


async def _ingest_fact(
    *,
    pool: Any,
    schema: str,
    item: NoesisInputItem,
    component_index: int,
    component: Any,
    data: dict,
    resolution: TimeResolution,
    identity_client: Any,
    identity_spec: IdentitySpec,
) -> tuple[int, dict[str, Any] | None]:
    """One fact, one short transaction.

    Returns ``(event_id, vector_alert)`` where ``vector_alert`` is the
    post-commit aggregate identity-vector alert payload (or None). Requirement
    03 §8.1 order: health/profile gates → atom precheck → sequential bge (all
    outside the transaction) → the requirement 02 short transaction with the
    vector-carrying upsert. 04A: the event is sequence-numbered via
    ``INSERT ... RETURNING event_id``; the returned id is shared with
    ``event_atoms`` inside the same transaction.
    """
    event_time = resolution.event_time

    # 1. Ordered unique E/P literals of this component (G never gets a vector).
    literals: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for atom in component.atoms:
        literal = (atom.text, atom.type)
        if atom.type in ("E", "P") and literal not in seen:
            seen.add(literal)
            literals.append(literal)

    # 2-4. Health gate, profile gate, atom-state precheck, sequential bge.
    prep = await _prepare_identity_vectors(
        pool=pool, schema=schema, client=identity_client, spec=identity_spec, literals=literals
    )

    # 5. The requirement 02 short transaction (unchanged boundaries).
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Fence the transaction against a rebuild that starts after the
            # transaction-external profile check and BGE calls. FOR SHARE is
            # held until commit, so a rebuild status UPDATE must wait. If the
            # rebuild/new generation already won, discard the prepared vectors
            # while preserving the fact-only degradation path.
            transaction_vectors = prep.vectors
            transaction_alert = prep.alert
            if transaction_vectors:
                profile_row = await conn.fetchrow(
                    _sql(schema, _PROFILE_WRITE_GUARD), _IDENTITY_KIND
                )
                db_profile = None
                if profile_row is not None:
                    db_profile = {
                        "model_name": profile_row["model_name"],
                        "model_revision": profile_row["model_revision"],
                        "dimension": int(profile_row["dimension"]),
                        "status": profile_row["status"],
                    }
                expected = (
                    identity_spec.model,
                    identity_spec.revision,
                    identity_spec.dimension,
                    "ready",
                )
                actual = None if db_profile is None else (
                    db_profile["model_name"],
                    db_profile["model_revision"],
                    db_profile["dimension"],
                    db_profile["status"],
                )
                if actual != expected:
                    transaction_vectors = {}
                    transaction_alert = _identity_alert(
                        stage="identity_vector",
                        alert_code="identity_vector_profile_mismatch",
                        severity="error",
                        message=(
                            "database identity profile changed before commit; "
                            "prepared vectors were discarded"
                        ),
                        spec=identity_spec,
                        db_profile=db_profile,
                    )
            row = await conn.fetchrow(
                _sql(schema, _EVENT_INSERT),
                event_time,
                json.dumps(data, ensure_ascii=False),
                "L",
                "R",
            )
            event_id = row["event_id"]
            for warning in resolution.warnings:
                await _write_alert(
                    conn,
                    schema,
                    dedupe_key=compute_alert_dedupe_key(
                        item=item,
                        stage="event_time",
                        alert_code=warning["alert_code"],
                        component_index=component_index,
                    ),
                    event_id=event_id,
                    stage="event_time",
                    alert_code=warning["alert_code"],
                    severity="warning",
                    message=warning["message"],
                    details=warning["details"],
                )

            atom_ids: dict[tuple[str, str], int] = {}
            for atom in component.atoms:
                literal = (atom.text, atom.type)
                if literal in atom_ids:
                    continue  # one upsert per typed literal per event
                vector = transaction_vectors.get(literal)
                atom_row = await conn.fetchrow(
                    _sql(schema, _ATOM_UPSERT),
                    atom.text,
                    atom.type,
                    _vector_literal(vector) if vector is not None else None,
                )
                atom_ids[literal] = atom_row["atom_id"]
            for atom in component.atoms:
                await conn.execute(
                    _sql(schema, _EVENT_ATOM_INSERT),
                    event_id,
                    atom.pos,
                    atom_ids[(atom.text, atom.type)],
                    _ROLE_TYPE[atom.role],
                    atom.target_occ,
                )
            return event_id, transaction_alert


# ---------------------------------------------------------------------------
# Routing (requirement 02 §8)
# ---------------------------------------------------------------------------

async def _route_component(
    *,
    pool: Any,
    schema: str,
    item: NoesisInputItem,
    component_index: int,
    component: Any,
    resolution: TimeResolution,
    attempts: int,
    identity_client: Any,
    identity_spec: IdentitySpec,
) -> None:
    component_json = component.model_dump(mode="json")
    data = _build_event_data(
        item=item, component_index=component_index, component_json=component_json, time_metadata=resolution.metadata
    )
    try:
        event_id, vector_alert = await _ingest_fact(
            pool=pool,
            schema=schema,
            item=item,
            component_index=component_index,
            component=component,
            data=data,
            resolution=resolution,
            identity_client=identity_client,
            identity_spec=identity_spec,
        )
    except Exception as error:
        await _write_alert_safe(
            pool,
            schema,
            item,
            stage="event_ingest",
            alert_code="event_ingest_failed",
            severity="error",
            message=f"noesis fact transaction failed: {type(error).__name__}",
            component_index=component_index,
            event_id=None,
            details=build_source_envelope(item=item, attempts=attempts),
        )
    else:
        # 7. Post-commit aggregate identity-vector alert (§8.1). The fact is
        # already committed; this alert's own failure only logs (§3.2.10).
        if vector_alert is not None:
            await _write_alert_safe(
                pool,
                schema,
                item,
                stage=vector_alert["stage"],
                alert_code=vector_alert["alert_code"],
                severity=vector_alert["severity"],
                message=vector_alert["message"],
                component_index=component_index,
                event_id=event_id,
                details={**build_source_envelope(item=item, attempts=attempts), **vector_alert["details"]},
            )


async def _ingest_item(
    *,
    item: NoesisInputItem,
    schema: str,
    timezone_name: str,
    extract_once: Callable[[str], object],
    analyzer: Any,
    pool_factory: Any,
    config: Any,
    identity_client: Any,
    identity_spec: IdentitySpec,
) -> None:
    """Process one item; never raises to the batch loop. Only business/dependency
    failures are isolated — cancellation propagates. (R02-04)"""
    try:
        try:
            outcome = await asyncio.to_thread(extract_noesis_components, item.content, extract_once=extract_once)
        except Exception as error:
            logger.error("noesis extraction crashed for item %s: %s", item.item_index, type(error).__name__)
            await _item_alert_safe(
                schema, item, config, pool_factory,
                stage="hyper_extract", alert_code="extraction_failed", severity="error",
                message=f"noesis extraction crashed: {type(error).__name__}", component_index=None,
                event_id=None, details=build_source_envelope(item=item, attempts=0),
            )
            return

        # Validate the outcome shape so an unexpected return type surfaces as an
        # observable per-item failure instead of an AttributeError mid-loop.
        if not hasattr(outcome, "components") or not hasattr(outcome, "alerts") or not hasattr(outcome, "attempts"):
            raise NoesisConfigError(f"unexpected extraction outcome type: {type(outcome).__name__}")

        pool = await _acquire_pool(config, pool_factory)
        envelope = build_source_envelope(item=item, attempts=outcome.attempts)
        for alert in outcome.alerts:
            await _write_alert_safe(
                pool, schema, item, stage=alert.stage, alert_code=alert.alert_code, severity=alert.severity,
                message=alert.message, component_index=None, event_id=None,
                details={**envelope, **(alert.details or {})},
            )
        for component_index, component in enumerate(outcome.components):
            if component.utterance_type == "hypothesis":
                await _write_alert_safe(
                    pool, schema, item, stage="hypothesis_routing", alert_code="hypothesis_classification_pending",
                    severity="info",
                    message=(
                        "hypothesis recorded as observability signal; classification interface "
                        "not yet available; component is not persisted as a rule or event"
                    ),
                    component_index=component_index, event_id=None,
                    details={**envelope, "component": component.model_dump(mode="json")},
                )
                continue
            resolution = resolve_event_time(
                observed_at=item.observed_at,
                atoms=component.atoms,
                analyzer=analyzer,
                timezone_name=timezone_name,
            )
            await _route_component(
                pool=pool, schema=schema, item=item, component_index=component_index,
                component=component, resolution=resolution, attempts=outcome.attempts,
                identity_client=identity_client, identity_spec=identity_spec,
            )
    except asyncio.CancelledError:
        raise  # never swallow cancellation semantics
    except Exception as error:
        logger.error("noesis item %s failed: %s", item.item_index, type(error).__name__)
        await _item_alert_safe(
            schema, item, config, pool_factory,
            stage="event_ingest", alert_code="event_ingest_failed", severity="error",
            message=f"noesis item processing failed: {type(error).__name__}", component_index=None,
            event_id=None, details=build_source_envelope(item=item, attempts=0),
        )


async def _item_alert_safe(
    schema: str,
    item: NoesisInputItem,
    config: Any,
    pool_factory: Any,
    **kwargs: Any,
) -> None:
    """Best-effort alert for a per-item failure; never raises, never recurses."""
    try:
        pool = await _acquire_pool(config, pool_factory)
        await _write_alert_safe(pool, schema, item, **kwargs)
    except Exception as error:
        logger.error(
            "noesis item alert could not be written (%s/%s): %s",
            kwargs.get("stage"), kwargs.get("alert_code"), type(error).__name__,
        )


async def ingest_noesis_batch(
    contents_dicts: Any,
    bank_id: str,
    config: Any,
    *,
    llm_config: Any = None,
    operation_id: str | None = None,
    document_id: str | None = None,
    analyzer: Any = None,
    extract_once_factory: Callable[[Any], Callable[[str], object]] | None = None,
    pool_factory: Callable[[Any], Any] | None = None,
    clock: Callable[[], datetime] | None = None,
    identity_client_factory: Callable[[Any], Any] | None = None,
) -> None:
    """Production entry point: never raises, never blocks the native retain."""
    if not getattr(config, "noesis_enabled", False):
        return  # deployment switch: a strict no-op, never a fallback to old chains

    schema = getattr(config, "noesis_schema", DEFAULT_NOESIS_SCHEMA) or DEFAULT_NOESIS_SCHEMA
    if not validate_schema_identifier(schema):
        logger.error("noesis ingest skipped: configured noesis_schema %r is not a valid SQL identifier", schema)
        return

    items = await normalize_items(
        contents_dicts,
        bank_id=bank_id,
        operation_id=operation_id,
        batch_document_id=document_id,
        clock=clock,
    )
    if not items:
        return

    identity_spec = _identity_spec_from_config(config)

    # Read-only schema preflight (once per process). If a required object is
    # missing we disable this process's Noesis ingestion (fail-closed to native
    # retain); a transient probe failure skips this batch and retries later.
    # Never ALTERs. The declared atoms.embedding width must match the
    # configured identity generation (requirement 03 §11).
    try:
        ready_pool = await _acquire_pool(config, pool_factory)
    except Exception as error:
        logger.error("noesis pool unavailable: %s", type(error).__name__)
        return
    if not await _ensure_schema_ready(ready_pool, schema, expected_dimension=identity_spec.dimension):
        return

    try:
        extract_once = (extract_once_factory or _production_extract_once_factory)(llm_config)
    except Exception as error:
        logger.error("noesis extraction client unavailable: %s", type(error).__name__)
        try:
            pool = await _acquire_pool(config, pool_factory)
            await _write_alert_safe(
                pool,
                schema,
                items[0],
                stage="noesis_config",
                alert_code="noesis_llm_config_invalid",
                severity="error",
                message=f"noesis extraction LLM configuration is unusable: {type(error).__name__}",
                component_index=None,
                event_id=None,
                details=build_source_envelope(item=items[0], attempts=0),
            )
        except Exception as pool_error:
            logger.error("noesis config alert could not be written: %s", type(pool_error).__name__)
        return

    try:
        if analyzer is None:
            from ..query_analyzer import DateparserQueryAnalyzer

            analyzer = DateparserQueryAnalyzer()
    except Exception as error:
        logger.error("noesis analyzer construction failed: %s", type(error).__name__)
        try:
            pool = await _acquire_pool(config, pool_factory)
            await _write_alert_safe(
                pool,
                schema,
                items[0],
                stage="noesis_config",
                alert_code="noesis_llm_config_invalid",
                severity="error",
                message=f"noesis time analyzer is unusable: {type(error).__name__}",
                component_index=None,
                event_id=None,
                details=build_source_envelope(item=items[0], attempts=0),
            )
        except Exception as pool_error:
            logger.error("noesis analyzer alert could not be written: %s", type(pool_error).__name__)
        return

    # One shared identity-vector client for the whole batch (lazy singleton in
    # production; factory seam in tests). Construction is trivial and must
    # never block retain — a failure here degrades to NULL vectors plus the
    # per-fact aggregate alert (requirement 03 §5.4/§9).
    try:
        identity_client = await _acquire_identity_client(config, identity_spec, identity_client_factory)
    except Exception as error:  # pragma: no cover - construction never fails in practice
        logger.error("noesis identity client unavailable: %s", type(error).__name__)
        identity_client = None

    timezone_name = getattr(config, "noesis_timezone", DEFAULT_NOESIS_TIMEZONE) or DEFAULT_NOESIS_TIMEZONE
    for item in items:
        await _ingest_item(
            item=item,
            schema=schema,
            timezone_name=timezone_name,
            extract_once=extract_once,
            analyzer=analyzer,
            pool_factory=pool_factory,
            config=config,
            identity_client=identity_client,
            identity_spec=identity_spec,
        )
