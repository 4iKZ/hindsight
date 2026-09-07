"""Shared offline fakes for the Noesis ingestion tests.

No network, no real PostgreSQL: every store interaction goes through a small
in-memory emulation of the four ``noesis_core`` statements the ingest module
is allowed to run (event insert/replay select, atom upsert, event_atoms
insert, alert insert). Transaction rollback is emulated with snapshots so the
failure-injection tests can prove zero-half-state.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from hyperextract.noesis import ExtractionAlert, ExtractionOutcome, FactComponent, HypothesisComponent


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
    """In-memory emulation of the four noesis_core statements the ingest runs."""

    def __init__(self) -> None:
        self.events: dict[tuple[str, datetime], dict[str, Any]] = {}
        self.atoms: dict[tuple[str, str], dict[str, int]] = {}
        self.event_atoms: list[tuple] = []
        self.alerts: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str, tuple]] = []
        self.commits = 0
        self.rollbacks = 0
        # marker -> number of matching SQL calls allowed before InjectedFailure
        self.fail_after: dict[str, int] = {}
        self.fail_on_commit = False
        self.interleave: Any = None  # optional async hook run before each SQL dispatch
        self._marker_counts: dict[str, int] = {}
        self._ids = {"event": 1000, "atom": 500, "alert": 9000}

    # -- SQL dispatch -------------------------------------------------------

    def _maybe_fail(self, sql: str) -> None:
        for marker, allowed in self.fail_after.items():
            if marker in sql:
                count = self._marker_counts.get(marker, 0) + 1
                self._marker_counts[marker] = count
                if count > allowed:
                    raise InjectedFailure(f"injected failure at marker {marker!r} (call #{count})")

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        if self.interleave is not None:
            await self.interleave()
        self._maybe_fail(sql)
        self.calls.append(("fetchrow", sql, args))
        upper = sql.lstrip().upper()
        if ".events" in sql and upper.startswith("INSERT"):
            return self._insert_event(args)
        if ".events" in sql and upper.startswith("SELECT"):
            return self._select_event(args)
        if ".atoms" in sql:
            return self._upsert_atom(args)
        if ".ingestion_alerts" in sql:
            return self._insert_alert(args)
        raise AssertionError(f"unexpected fetchrow SQL: {sql}")

    async def execute(self, sql: str, *args: Any) -> str:
        self._maybe_fail(sql)
        self.calls.append(("execute", sql, args))
        if ".event_atoms" in sql:
            self.event_atoms.append(args)
            return "INSERT 0 1"
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

    def _upsert_atom(self, args: tuple) -> dict[str, Any]:
        text, atom_type = args[0], args[1]
        current = self.atoms.get((text, atom_type))
        if current is None:
            self._ids["atom"] += 1
            self.atoms[(text, atom_type)] = {"atom_id": self._ids["atom"], "support_count": 1}
        else:
            current["support_count"] += 1
        return {"atom_id": self.atoms[(text, atom_type)]["atom_id"]}

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


class _FakeTransaction:
    """``async with conn.transaction():`` emulation with snapshot rollback."""

    def __init__(self, store: FakeStore) -> None:
        self._store = store

    async def __aenter__(self) -> None:
        self._snapshot = copy.deepcopy(
            (self._store.events, self._store.atoms, self._store.event_atoms, self._store.alerts, self._store._ids)
        )
        self._marker_counts = dict(self._store._marker_counts)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self._rollback()
            return False
        if self._store.fail_on_commit:
            self._rollback()
            raise InjectedFailure("injected failure at commit")
        self._store.commits += 1
        return False

    def _rollback(self) -> None:
        events, atoms, event_atoms, alerts, ids = self._snapshot
        self._store.events = events
        self._store.atoms = atoms
        self._store.event_atoms = event_atoms
        self._store.alerts = alerts
        self._store._ids = ids
        self._store._marker_counts = self._marker_counts
        self._store.rollbacks += 1


class FakeConn:
    def __init__(self, store: FakeStore) -> None:
        self._store = store

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self._store)

    async def fetchrow(self, sql: str, *args: Any):
        return await self._store.fetchrow(sql, *args)

    async def fetchval(self, sql: str, *args: Any):
        # The ingestion preflight is read-only; the in-memory store models the
        # fully migrated schema and therefore answers every catalog probe true.
        return True

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
