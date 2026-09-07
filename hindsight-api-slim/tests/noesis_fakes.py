"""Shared offline fakes for the Noesis ingestion tests.

No network, no real PostgreSQL: every store interaction goes through a small
in-memory emulation of the ``noesis_core`` statements the ingest module is
allowed to run (event insert/replay select, atom state precheck, atom upsert,
event_atoms insert, alert insert, embedding profile gate). Transaction
rollback is emulated with snapshots so the failure-injection tests can prove
zero-half-state. Requirement 03 adds the identity-vector seams:
``FakeIdentityClient`` (bge stand-in) and the embedding/profile emulation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from hyperextract.noesis import ExtractionAlert, ExtractionOutcome, FactComponent, HypothesisComponent

from hindsight_api.engine.retain.noesis_identity_vector import (
    IdentityConfigInvalid,
    IdentityEmbedError,
    IdentityServiceUnavailable,
)


class InjectedFailure(RuntimeError):
    """Raised by the fake store at a scripted failure point."""


# ---------------------------------------------------------------------------
# Golden components (verbatim from requirement 01 examples / requirement 02 §3.4.9)
# ---------------------------------------------------------------------------

GOLDEN_FACT_TIME_JSON = {
    "utterance_type": "fact",
    "atoms": [
        {"pos": 1, "text": "昨天", "type": "E", "role": "modifier", "target_occ": 4, "resolved": None},
        {"pos": 2, "text": "妈妈", "type": "E", "role": "agent", "target_occ": 4, "resolved": None},
        {"pos": 3, "text": "在超市", "type": "E", "role": "modifier", "target_occ": 4, "resolved": None},
        {"pos": 4, "text": "买", "type": "P", "role": "predicate", "target_occ": None, "resolved": None},
        {"pos": 5, "text": "苹果", "type": "E", "role": "patient", "target_occ": 4, "resolved": None},
    ],
    "tree": {
        "predicate": "买",
        "agent": [{"text": "妈妈", "modifier": [], "implied": False}],
        "patient": [{"text": "苹果", "modifier": [], "implied": False}],
        "modifier": ["昨天", "在超市"],
        "nested": [],
        "conditional": [],
    },
}

GOLDEN_HYPOTHESIS_JSON = {
    "utterance_type": "hypothesis",
    "atoms": [
        {"pos": 1, "text": "太阳", "type": "E", "role": "agent", "target_occ": 3, "resolved": None},
        {"pos": 2, "text": "每天", "type": "E", "role": "modifier", "target_occ": 3, "resolved": None},
        {"pos": 3, "text": "升起", "type": "P", "role": "predicate", "target_occ": None, "resolved": None},
        {"pos": 4, "text": "东边", "type": "E", "role": "modifier", "target_occ": 3, "resolved": None},
    ],
    "tree": {
        "predicate": "升起",
        "agent": [{"text": "太阳", "modifier": [], "implied": False}],
        "patient": [],
        "modifier": ["每天", "东边"],
        "nested": [],
        "conditional": [],
    },
    "rule_template": {
        "premise": [{"text": "太阳", "type": "E", "role": "agent"}],
        "conclusion": {"predicate": "升起", "agent": ["太阳"], "patient": [], "modifier": ["东边"]},
        "condition": ["每天"],
    },
}

GOLDEN_FACT_RECURSIVE_JSON = {
    "utterance_type": "fact",
    "atoms": [
        {"pos": 1, "text": "小明", "type": "E", "role": "agent", "target_occ": 4, "resolved": None},
        {"pos": 2, "text": "没写", "type": "P", "role": "predicate", "target_occ": 4, "resolved": None},
        {"pos": 3, "text": "作业", "type": "E", "role": "patient", "target_occ": 2, "resolved": None},
        {"pos": 4, "text": "揍", "type": "P", "role": "predicate", "target_occ": None, "resolved": None},
        {"pos": 5, "text": "小明", "type": "E", "role": "patient", "target_occ": 4, "resolved": True},
    ],
    "tree": {
        "predicate": "揍",
        "agent": [{"text": "小明", "modifier": [], "implied": False}],
        "patient": [{"text": "小明", "modifier": [], "implied": False}],
        "modifier": [],
        "nested": [
            {
                "predicate": "没写",
                "agent": [{"text": "小明", "modifier": [], "implied": True}],
                "patient": [{"text": "作业", "modifier": [], "implied": False}],
                "modifier": [],
                "nested": [],
                "conditional": [],
            }
        ],
        "conditional": [],
    },
}


def golden_fact_time() -> FactComponent:
    return FactComponent.model_validate(GOLDEN_FACT_TIME_JSON)


def golden_fact_recursive() -> FactComponent:
    return FactComponent.model_validate(GOLDEN_FACT_RECURSIVE_JSON)


def golden_hypothesis() -> HypothesisComponent:
    return HypothesisComponent.model_validate(GOLDEN_HYPOTHESIS_JSON)


def he_alert(code: str = "invalid_component_dropped", severity: str = "warning") -> ExtractionAlert:
    return ExtractionAlert(
        stage="hyper_extract",
        alert_code=code,
        severity=severity,
        message="fake extraction alert",
        details={"component_index": 0, "rule": "closure"},
    )


def outcome(components=(), alerts=(), attempts: int = 1) -> ExtractionOutcome:
    return ExtractionOutcome(components=list(components), alerts=list(alerts), attempts=attempts)


# ---------------------------------------------------------------------------
# Config / LLM config fakes
# ---------------------------------------------------------------------------

def noesis_config(**overrides: Any) -> SimpleNamespace:
    cfg = SimpleNamespace(
        noesis_enabled=True,
        noesis_database_url="postgresql://noesis-test-user@localhost:5432/noesis",
        noesis_schema="noesis_core",
        noesis_timezone="Asia/Shanghai",
        noesis_pool_min_size=1,
        noesis_pool_max_size=5,
        noesis_command_timeout=10,
        # Requirement 03 embedding fields: deliberately NOT the production
        # defaults, so a default leak would be visible in assertions.
        noesis_embedding_base_url="http://identity.test",
        noesis_embedding_model="bge-m3",
        noesis_embedding_revision="bge-m3-1024-v1",
        noesis_embedding_dimension=1024,
        noesis_embedding_timeout_seconds=2.0,
        noesis_embedding_max_retries=1,
        noesis_embedding_api_key="",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def llm_config(provider: str = "openai", model: str = "test-model") -> SimpleNamespace:
    return SimpleNamespace(provider=provider, model=model, base_url="http://llm.test/v1", api_key="sk-test")


# ---------------------------------------------------------------------------
# Fake asyncpg pool / connection / store
# ---------------------------------------------------------------------------

class FakeStore:
    """In-memory emulation of the noesis_core statements the ingest runs."""

    def __init__(self) -> None:
        self.events: dict[tuple[str, datetime], dict[str, Any]] = {}
        self.atoms: dict[tuple[str, str], dict[str, Any]] = {}
        self.event_atoms: list[tuple] = []
        self.alerts: list[dict[str, Any]] = []
        # Requirement 03: the single-row identity embedding profile gate.
        self.profile: dict[str, Any] | None = None
        self.identity_rebuild_stage: dict[int, tuple[float, ...]] = {}
        self.calls: list[tuple[str, str, tuple]] = []
        self.tx_log: list[str] = []  # "begin" / "commit" / "rollback" markers
        self.commits = 0
        self.rollbacks = 0
        # marker -> number of matching SQL calls allowed before InjectedFailure.
        # Failure injection fires on in-transaction statements and on writes;
        # pre-transaction read-only probes (replay select, atom state precheck)
        # never carry business half-state, so failing them is outside the
        # zero-half-state contract these markers exist to prove.
        self.fail_after: dict[str, int] = {}
        self.fail_on_commit = False
        self.interleave: Any = None  # async hook run before each in-tx SQL dispatch
        self.embedding_dimension = 1024  # the fake pgvector column width
        # Rebuild-command knobs (requirement 03 §10.4 tests).
        self.advisory_lock_available = True
        self.pg_indexes: list[dict[str, Any]] = []
        self._marker_counts: dict[str, int] = {}
        self._ids = {"event": 1000, "atom": 500, "alert": 9000}
        self._tx_depth = 0

    # -- SQL dispatch -------------------------------------------------------

    def _maybe_fail(self, sql: str) -> None:
        # Gated to transactional statements and to writes: pre-transaction
        # read-only probes (replay select, atom state precheck) never carry
        # business half-state, so failing them is outside the zero-half-state
        # contract these markers exist to prove. Post-commit alert INSERTs
        # run outside a transaction yet stay fail-able (they are writes).
        if self._tx_depth == 0 and not sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            return
        for marker, allowed in self.fail_after.items():
            if marker in sql:
                count = self._marker_counts.get(marker, 0) + 1
                self._marker_counts[marker] = count
                if count > allowed:
                    raise InjectedFailure(f"injected failure at marker {marker!r} (call #{count})")

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        if self.interleave is not None and self._tx_depth > 0:
            await self.interleave()
        self._maybe_fail(sql)
        self.calls.append(("fetchrow", sql, args))
        upper = sql.lstrip().upper()
        if ".embedding_profiles" in sql:
            return self._profile_dispatch(upper, args)
        if ".events" in sql and upper.startswith("INSERT"):
            return self._insert_event(args)
        if ".events" in sql and upper.startswith("SELECT"):
            return self._select_event(args)
        if ".atoms" in sql:
            return self._upsert_atom(args)
        if ".ingestion_alerts" in sql:
            return self._insert_alert(args)
        raise AssertionError(f"unexpected fetchrow SQL: {sql}")

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self._maybe_fail(sql)
        self.calls.append(("fetch", sql, args))
        if "unnest" in sql:
            texts, atom_types = args[0], args[1]
            return [self._atom_state_row(text, atom_type) for text, atom_type in zip(texts, atom_types)]
        if "pg_indexes" in sql:
            return list(self.pg_indexes)
        if "LEFT JOIN pg_temp.noesis_identity_rebuild_stage" in sql:
            null_only = "a.embedding IS NULL" in sql
            rows = []
            for (text, atom_type), atom in sorted(self.atoms.items(), key=lambda kv: kv[1]["atom_id"]):
                if atom_type not in ("E", "P") or atom["atom_id"] in self.identity_rebuild_stage:
                    continue
                if null_only and atom["embedding"] is not None:
                    continue
                rows.append({"atom_id": atom["atom_id"], "text": text, "atom_type": atom_type})
            return rows
        if ".atoms" in sql and "ORDER BY atom_id" in sql:
            null_only = "embedding IS NULL" in sql
            rows = []
            for (text, atom_type), atom in sorted(self.atoms.items(), key=lambda kv: kv[1]["atom_id"]):
                if atom_type not in ("E", "P"):
                    continue
                if null_only and atom["embedding"] is not None:
                    continue
                rows.append({"atom_id": atom["atom_id"], "text": text, "atom_type": atom_type})
            return rows
        raise AssertionError(f"unexpected fetch SQL: {sql}")

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.calls.append(("fetchval", sql, args))
        if "format_type" in sql:
            # The fake catalog models the fully migrated VECTOR(1024) column.
            return f"vector({self.embedding_dimension})"
        if "pg_try_advisory_lock" in sql:
            return self.advisory_lock_available
        if "embedding IS NOT NULL" in sql and "count(*)" in sql:
            return sum(
                1
                for (text, atom_type), atom in self.atoms.items()
                if atom_type in ("E", "P") and atom["embedding"] is not None
            )
        # The ingestion preflight is read-only; the in-memory store models the
        # fully migrated schema and therefore answers every catalog probe true.
        return True

    async def execute(self, sql: str, *args: Any) -> str:
        self._maybe_fail(sql)
        self.calls.append(("execute", sql, args))
        if ".event_atoms" in sql:
            self.event_atoms.append(args)
            return "INSERT 0 1"
        if "pg_advisory_unlock" in sql:
            return "SELECT 1"
        if sql.startswith("CREATE TEMP TABLE"):
            return "CREATE TABLE"
        if sql.startswith("TRUNCATE TABLE pg_temp.noesis_identity_rebuild_stage"):
            self.identity_rebuild_stage.clear()
            return "TRUNCATE TABLE"
        if sql.startswith("DROP TABLE IF EXISTS pg_temp.noesis_identity_rebuild_stage"):
            self.identity_rebuild_stage.clear()
            return "DROP TABLE"
        if sql.startswith("INSERT INTO pg_temp.noesis_identity_rebuild_stage"):
            self.identity_rebuild_stage[args[0]] = _parse_vector_literal(args[1])
            return "INSERT 0 1"
        if sql.startswith("LOCK TABLE"):
            return "LOCK TABLE"
        if ".embedding_profiles" in sql and "SET status = 'rebuilding'" in sql:
            if self.profile is not None:
                self.profile["status"] = "rebuilding"
            return "UPDATE 1"
        if ".embedding_profiles" in sql and "SET model_name" in sql:
            if self.profile is not None:
                self.profile.update(
                    {"model_name": args[0], "model_revision": args[1], "dimension": args[2], "status": "ready"}
                )
            return "UPDATE 1"
        if ".atoms" in sql and "FROM pg_temp.noesis_identity_rebuild_stage" in sql:
            null_guard = "a.embedding IS NULL" in sql
            updated = 0
            for atom in self.atoms.values():
                vector = self.identity_rebuild_stage.get(atom["atom_id"])
                if vector is None or (null_guard and atom["embedding"] is not None):
                    continue
                atom["embedding"] = vector
                updated += 1
            return f"UPDATE {updated}"
        if ".atoms" in sql and "SET embedding" in sql:
            null_guard = "AND embedding IS NULL" in sql
            for atom in self.atoms.values():
                if atom["atom_id"] == args[0]:
                    if not (null_guard and atom["embedding"] is not None):
                        atom["embedding"] = _parse_vector_literal(args[1])
                    return "UPDATE 1"
            return "UPDATE 0"
        raise AssertionError(f"unexpected execute SQL: {sql}")

    # -- table emulation ----------------------------------------------------

    def _insert_event(self, args: tuple) -> dict[str, Any] | None:
        event_time, ingestion_key, data_json = args[0], args[1], args[2]
        key = (ingestion_key, event_time)
        if key in self.events:
            return None
        self._ids["event"] += 1
        self.events[key] = {
            "event_id": self._ids["event"],
            "event_time": event_time,
            "data": json.loads(data_json),
        }
        return {"event_id": self._ids["event"]}

    def _select_event(self, args: tuple) -> dict[str, Any] | None:
        row = self.events.get((args[0], args[1]))
        if row is None:
            return None
        return {"event_id": row["event_id"], "data": json.dumps(row["data"])}

    def _atom_state_row(self, text: str, atom_type: str) -> dict[str, Any]:
        atom = self.atoms.get((text, atom_type))
        return {
            "text": text,
            "atom_type": atom_type,
            "atom_exists": atom is not None,
            "has_embedding": atom is not None and atom["embedding"] is not None,
        }

    def _upsert_atom(self, args: tuple) -> dict[str, Any]:
        text, atom_type, vector = args[0], args[1], _parse_vector_literal(args[2])
        if vector is not None and len(vector) != self.embedding_dimension:
            # pgvector rejects wrong-width vectors at write time (§8.4).
            raise InjectedFailure(f"pgvector dimension reject: {len(vector)} != {self.embedding_dimension}")
        current = self.atoms.get((text, atom_type))
        if current is None:
            self._ids["atom"] += 1
            self.atoms[(text, atom_type)] = {
                "atom_id": self._ids["atom"],
                "support_count": 1,
                "embedding": vector,
            }
        else:
            current["support_count"] += 1
            # CASE branch of the frozen upsert: backfill only, never overwrite.
            if current["embedding"] is None and vector is not None:
                current["embedding"] = vector
        return {"atom_id": self.atoms[(text, atom_type)]["atom_id"]}

    def _profile_dispatch(self, upper: str, args: tuple) -> dict[str, Any] | None:
        if upper.startswith("SELECT"):
            return None if self.profile is None else dict(self.profile)
        if upper.startswith("INSERT"):
            if self.profile is not None:
                return None  # ON CONFLICT (embedding_kind) DO NOTHING
            self.profile = {
                "embedding_kind": "identity",
                "model_name": args[0],
                "model_revision": args[1],
                "dimension": args[2],
                "status": "ready",
            }
            return {"embedding_kind": "identity"}
        raise AssertionError(f"unexpected embedding_profiles SQL dispatch: {upper}")

    def _insert_alert(self, args: tuple) -> dict[str, Any] | None:
        dedupe_key = args[0]
        if any(alert["dedupe_key"] == dedupe_key for alert in self.alerts):
            return None
        self._ids["alert"] += 1
        alert = {
            "alert_id": self._ids["alert"],
            "dedupe_key": dedupe_key,
            "event_id": args[1],
            "stage": args[2],
            "alert_code": args[3],
            "severity": args[4],
            "message": args[5],
            "details": json.loads(args[6]),
        }
        self.alerts.append(alert)
        return {"alert_id": alert["alert_id"]}

    # -- helpers ------------------------------------------------------------

    def event_rows(self) -> list[dict[str, Any]]:
        return list(self.events.values())

    def alerts_by_code(self, code: str) -> list[dict[str, Any]]:
        return [alert for alert in self.alerts if alert["alert_code"] == code]

    def support_count(self, text: str, atom_type: str) -> int:
        return self.atoms.get((text, atom_type), {}).get("support_count", 0)

    def embedding_of(self, text: str, atom_type: str) -> tuple | None:
        return self.atoms.get((text, atom_type), {}).get("embedding")

    def fetch_calls(self, marker: str) -> list[tuple]:
        return [call for call in self.calls if marker in call[1]]


def _parse_vector_literal(literal: str | None) -> tuple[float, ...] | None:
    """Parse the ``$3::vector`` text literal back into floats (fake pgvector)."""
    if literal is None:
        return None
    body = literal.strip()
    assert body.startswith("[") and body.endswith("]"), f"not a vector literal: {literal!r}"
    return tuple(float(component) for component in body[1:-1].split(","))


class _FakeTransaction:
    """``async with conn.transaction():`` emulation with snapshot rollback."""

    def __init__(self, store: FakeStore) -> None:
        self._store = store

    async def __aenter__(self) -> None:
        self._store._tx_depth += 1
        self._store.tx_log.append("begin")
        self._snapshot = copy.deepcopy(
            (
                self._store.events,
                self._store.atoms,
                self._store.event_atoms,
                self._store.alerts,
                self._store.profile,
                self._store.identity_rebuild_stage,
                self._store._ids,
            )
        )
        self._marker_counts = dict(self._store._marker_counts)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self._store._tx_depth -= 1
        if exc_type is not None:
            self._rollback()
            return False
        if self._store.fail_on_commit:
            self._rollback()
            raise InjectedFailure("injected failure at commit")
        self._store.commits += 1
        self._store.tx_log.append("commit")
        return False

    def _rollback(self) -> None:
        events, atoms, event_atoms, alerts, profile, rebuild_stage, ids = self._snapshot
        self._store.events = events
        self._store.atoms = atoms
        self._store.event_atoms = event_atoms
        self._store.alerts = alerts
        self._store.profile = profile
        self._store.identity_rebuild_stage = rebuild_stage
        self._store._ids = ids
        self._store._marker_counts = self._marker_counts
        self._store.rollbacks += 1
        self._store.tx_log.append("rollback")


class FakeConn:
    def __init__(self, store: FakeStore) -> None:
        self._store = store

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self._store)

    async def fetchrow(self, sql: str, *args: Any):
        return await self._store.fetchrow(sql, *args)

    async def fetch(self, sql: str, *args: Any):
        return await self._store.fetch(sql, *args)

    async def fetchval(self, sql: str, *args: Any):
        return await self._store.fetchval(sql, *args)

    async def execute(self, sql: str, *args: Any):
        return await self._store.execute(sql, *args)


class _AcquireCtx:
    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> FakeConn:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class FakePool:
    def __init__(self, store: FakeStore) -> None:
        self.store = store
        self.acquire_count = 0
        self.closed = False

    def acquire(self) -> _AcquireCtx:
        self.acquire_count += 1
        return _AcquireCtx(FakeConn(self.store))

    def is_closed(self) -> bool:
        return self.closed

    async def close(self) -> None:
        self.closed = True


def pool_factory_for(store: FakeStore):
    """``pool_factory`` seam returning a FakePool over ``store``."""

    async def factory(_config):
        return FakePool(store)

    return factory


# ---------------------------------------------------------------------------
# Extraction seams
# ---------------------------------------------------------------------------

class FakeExtractOnceFactory:
    """``extract_once_factory`` seam: records the llm_config, serves raw payloads."""

    def __init__(self, raw_payloads: list | None = None, error: Exception | None = None) -> None:
        self.requested_llm_configs: list = []
        self.texts: list[str] = []
        self._raw_payloads = list(raw_payloads or [])
        self._error = error

    def __call__(self, llm_config):
        self.requested_llm_configs.append(llm_config)

        def extract_once(text: str) -> object:
            self.texts.append(text)
            if self._error is not None:
                raise self._error
            if self._raw_payloads:
                return self._raw_payloads.pop(0)
            return []

        return extract_once


# ---------------------------------------------------------------------------
# Identity-vector seams (requirement 03)
# ---------------------------------------------------------------------------

def fake_identity_vector(text: str, atom_type: str, dimension: int = 1024) -> tuple[float, ...]:
    """Deterministic stand-in for BGE(literal): sha256-seeded floats in [-1, 1)."""
    digest = hashlib.sha256(f"{atom_type}::{text}".encode("utf-8")).digest()
    return tuple(round((digest[index % len(digest)] / 255.0) * 2 - 1, 6) for index in range(dimension))


class FakeIdentityClient:
    """``identity_client_factory`` seam: scriptable bge stand-in.

    ``health`` is "ready" (default), "config_invalid", or "unavailable";
    ``failures`` maps (text, atom_type) -> error_kind for per-literal failures.
    """

    def __init__(
        self,
        *,
        health: str = "ready",
        failures: dict[tuple[str, str], str] | None = None,
        dimension: int = 1024,
    ) -> None:
        self.health = health
        self.failures = dict(failures or {})
        self.dimension = dimension
        self.ensure_calls = 0
        self.embed_calls: list[tuple[str, str]] = []
        self.closed = False

    async def ensure_ready(self) -> None:
        self.ensure_calls += 1
        if self.health == "config_invalid":
            raise IdentityConfigInvalid("fake /health config mismatch")
        if self.health == "unavailable":
            raise IdentityServiceUnavailable("fake /health transport failure")

    async def embed(self, text: str, atom_type: str) -> list[float]:
        self.embed_calls.append((text, atom_type))
        error_kind = self.failures.get((text, atom_type))
        if error_kind is not None:
            raise IdentityEmbedError(error_kind, f"fake embed failure: {error_kind}")
        return list(fake_identity_vector(text, atom_type, self.dimension))

    async def aclose(self) -> None:
        self.closed = True


def identity_factory_for(client: FakeIdentityClient):
    """``identity_client_factory`` seam returning ``client`` for any config."""

    async def factory(_config):
        return client

    return factory


# ---------------------------------------------------------------------------
# Fake time analyzer
# ---------------------------------------------------------------------------

class FakeAnalyzer:
    """Maps modifier text -> TemporalConstraint | None, or raises."""

    def __init__(self, mapping: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        from hindsight_api.engine.query_analyzer import TemporalConstraint

        self._temporal_constraint = TemporalConstraint
        self.mapping = mapping or {}
        self.error = error
        self.calls: list[tuple[str, datetime | None]] = []

    def analyze(self, query: str, reference_date: datetime | None = None):
        self.calls.append((query, reference_date))
        if self.error is not None:
            raise self.error
        return self.mapping.get(query)


def constraint(start: datetime, end: datetime | None = None):
    from hindsight_api.engine.query_analyzer import TemporalConstraint

    return TemporalConstraint(start_date=start, end_date=end or start + timedelta(days=1) - timedelta(microseconds=1))


# ---------------------------------------------------------------------------
# Common observed_at / content values
# ---------------------------------------------------------------------------

# 2026-09-05T10:00:00+08:00 — the requirement 02 §9.3 example observed_at.
OBSERVED_AT = datetime(2026, 9, 5, 2, 0, 0, tzinfo=UTC)
CONTENT = "小明没写作业后揍了自己。"
