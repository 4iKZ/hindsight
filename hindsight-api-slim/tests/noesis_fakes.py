"""Shared offline fakes for the Noesis ingestion tests.

No network, no real PostgreSQL: every store interaction goes through a small
in-memory emulation of the ``noesis_core`` statements the ingest module is
allowed to run (sequence-numbered event insert with RETURNING, atom state
precheck, atom upsert, event_atoms insert, alert insert, embedding profile
gate). Transaction rollback is emulated with snapshots so the failure-injection
tests can prove zero-half-state. Requirement 03 adds the identity-vector
embedding/profile emulation; requirement 05 adds the anchors emulation (atom
row locks, nearest-active cosine query, insert-with-count-1, EMA update,
transaction snapshots) and the unified ``FakeEmbeddingClient`` bge stand-in
(``embed_identity`` + ``embed_context``). Requirement 06 adds the two
write-maintenance bitmap tables (set-based ``rb64_build``/``rb64_or`` upserts
with ON CONFLICT semantics and rollback snapshots). Requirement 07 adds the
database-side ``rb64_andnot`` expired-window prune simulation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from hyperextract.noesis import ExtractionAlert, ExtractionOutcome, FactComponent, HypothesisComponent

from hindsight_api.engine.retain.noesis_embedding import (
    EmbeddingConfigInvalid,
    EmbeddingError,
    EmbeddingServiceUnavailable,
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


GOLDEN_FACT_PIVOT_JSON = {
    "utterance_type": "fact",
    "atoms": [
        {"pos": 1, "text": "妈妈", "type": "E", "role": "agent", "target_occ": 2, "resolved": None},
        {"pos": 2, "text": "让", "type": "P", "role": "predicate", "target_occ": None, "resolved": None},
        {"pos": 3, "text": "小明", "type": "E", "role": "patient", "target_occ": 2, "resolved": None},
        {"pos": 4, "text": "打", "type": "P", "role": "predicate", "target_occ": 3, "resolved": None},
        {"pos": 5, "text": "酱油", "type": "E", "role": "patient", "target_occ": 4, "resolved": None},
    ],
    "tree": {
        "predicate": "让",
        "agent": [{"text": "妈妈", "modifier": [], "implied": False}],
        "patient": [{"text": "小明", "modifier": [], "implied": False}],
        "modifier": [],
        "nested": [
            {
                "predicate": "打",
                "agent": [{"text": "小明", "modifier": [], "implied": True}],
                "patient": [{"text": "酱油", "modifier": [], "implied": False}],
                "modifier": [],
                "nested": [],
                "conditional": [],
            }
        ],
        "conditional": [],
    },
}


def golden_fact_recursive() -> FactComponent:
    return FactComponent.model_validate(GOLDEN_FACT_RECURSIVE_JSON)


def golden_fact_pivot() -> FactComponent:
    return FactComponent.model_validate(GOLDEN_FACT_PIVOT_JSON)


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
        # Requirement 04A: events have no ingestion_key and no unique replay
        # constraint — every insert gets a fresh sequence-numbered event_id.
        self.events: dict[int, dict[str, Any]] = {}
        self.atoms: dict[tuple[str, str], dict[str, Any]] = {}
        self.event_atoms: list[tuple] = []
        self.alerts: list[dict[str, Any]] = []
        # Requirement 03: the single-row identity embedding profile gate.
        self.profile: dict[str, Any] | None = None
        # Requirement 05A: the four rebuild TEMP staging tables (session-scoped).
        self.embedding_atom_stage: dict[int, tuple[float, ...]] = {}
        self.embedding_context_stage: dict[tuple[int, int], tuple[float, ...]] = {}
        self.embedding_anchor_sample_stage: dict[tuple[int, int, int], bool] = {}
        self.embedding_anchor_stage: dict[int, tuple[tuple[float, ...], int]] = {}
        # Requirement 05: anchors keyed by anchor_id; locked_atoms records the
        # FOR UPDATE atom row locks taken inside the fact transaction.
        self.anchors: dict[int, dict[str, Any]] = {}
        self.locked_atoms: list[int] = []
        # Requirement 06: the two write-maintenance bitmap tables, emulating
        # rb64_build (set construction) + rb64_or (union) on ON CONFLICT.
        self.cooccurrence_bitmaps: dict[tuple[int, int], set[int]] = {}
        self.neighbor_bitmaps: dict[tuple[int, int, str], set[int]] = {}
        self.calls: list[tuple[str, str, tuple]] = []
        self.tx_log: list[str] = []  # "begin" / "commit" / "rollback" markers
        self.commits = 0
        self.rollbacks = 0
        # marker -> number of matching SQL calls allowed before InjectedFailure.
        # Failure injection fires on in-transaction statements and on writes;
        # pre-transaction read-only probes (atom state precheck) never carry
        # business half-state, so failing them is outside the zero-half-state
        # contract these markers exist to prove.
        self.fail_after: dict[str, int] = {}
        self.fail_on_commit = False
        self.interleave: Any = None  # async hook run before each in-tx SQL dispatch
        self.embedding_dimension = 1024  # the fake pgvector column width
        # Rebuild-command knobs (requirement 03 §10.4 tests).
        self.advisory_lock_available = True
        self.pg_indexes: list[dict[str, Any]] = []
        self._marker_counts: dict[str, int] = {}
        self._ids = {"event": 1000, "atom": 500, "alert": 9000, "anchor": 100}
        self._tx_depth = 0

    # -- SQL dispatch -------------------------------------------------------

    def _maybe_fail(self, sql: str) -> None:
        # Gated to transactional statements and to writes: pre-transaction
        # read-only probes (atom state precheck) never carry business
        # half-state, so failing them is outside the zero-half-state
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
        # Requirement 05 dispatches. The atom row lock must precede the generic
        # ".atoms" branch or the single-arg lock SQL would be misrouted into
        # the three-arg upsert unpack.
        if "FOR UPDATE" in sql and ".atoms" in sql:
            return self._lock_atom(args)
        if ".anchors" in sql and upper.startswith("INSERT"):
            return self._insert_anchor(args)
        if ".anchors" in sql and "<=>" in sql:
            return self._nearest_anchor(args)
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
        # Requirement 05A rebuild: keyset-paginated atom/event scans (256 rows).
        if ".atoms" in sql and "atom_id > $1" in sql:
            null_only = "embedding IS NULL" in sql
            rows = []
            for (text, atom_type), atom in sorted(self.atoms.items(), key=lambda kv: kv[1]["atom_id"]):
                if atom_type not in ("E", "P") or atom.get("status", "active") != "active":
                    continue
                if int(atom["atom_id"]) <= args[0]:
                    continue
                if null_only and atom["embedding"] is not None:
                    continue
                rows.append({"atom_id": atom["atom_id"], "text": text, "atom_type": atom_type})
            return rows[:256]
        if ".events" in sql and "event_id > $1" in sql:
            rows = []
            for event_id, event in sorted(self.events.items()):
                if int(event_id) <= args[0]:
                    continue
                rows.append({"event_id": int(event_id), "data": event["data"]})
            return rows[:256]
        if ".event_atoms" in sql and "event_id = $1" in sql:
            rows = []
            for row in self.event_atoms:
                # (event_id, occurrence_id, atom_id, role_type, target_occ, anchor_id)
                if row[0] != args[0]:
                    continue
                rows.append(
                    {
                        "occurrence_id": row[1],
                        "atom_id": row[2],
                        "anchor_id": row[5] if len(row) > 5 else None,
                    }
                )
            return sorted(rows, key=lambda item: item["occurrence_id"])
        if ".atoms" in sql and "ANY(" in sql:
            wanted = set(args[0])
            return [
                {"atom_id": atom["atom_id"], "text": text, "atom_type": atom_type}
                for (text, atom_type), atom in self.atoms.items()
                if int(atom["atom_id"]) in wanted
            ]
        if ".anchors" in sql and "ANY(" in sql:
            wanted = set(args[0])
            return [
                {"anchor_id": anchor_id, "atom_id": anchor["atom_id"]}
                for anchor_id, anchor in self.anchors.items()
                if anchor_id in wanted
            ]
        if "noesis_embedding_anchor_stage" in sql and "LEFT JOIN" in sql:
            staged = self.embedding_anchor_stage
            rows = []
            for anchor_id, anchor in sorted(self.anchors.items()):
                if anchor_id in staged:
                    continue
                rows.append({"anchor_id": anchor_id, "atom_id": anchor["atom_id"]})
            return rows[:50]
        if "noesis_embedding_anchor_sample_stage" in sql and "JOIN" in sql:
            rows = []
            last_key = tuple(int(value) for value in args)
            for (anchor_id, event_id, predicate_pos) in sorted(self.embedding_anchor_sample_stage):
                if (anchor_id, event_id, predicate_pos) <= last_key:
                    continue
                context = self.embedding_context_stage.get((event_id, predicate_pos))
                if context is None:
                    continue
                rows.append(
                    {
                        "anchor_id": anchor_id,
                        "event_id": event_id,
                        "predicate_pos": predicate_pos,
                        "context_text": _format_vector_literal(context),
                    }
                )
            return rows[:256]
        raise AssertionError(f"unexpected fetch SQL: {sql}")

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.calls.append(("fetchval", sql, args))
        if ".cooccurrence_bitmaps" in sql and "rb64_andnot" in sql:
            # Requirement 07: expired-window ANDNOT prune. Only buckets whose
            # bitmap intersects the expired IDs are updated, emptied rows are
            # kept (never deleted), and the neighbor bitmaps are never touched.
            if self.interleave is not None and self._tx_depth > 0:
                await self.interleave()
            self._maybe_fail(sql)
            expired = set(args[0])
            updated = 0
            for key, bits in self.cooccurrence_bitmaps.items():
                if bits.intersection(expired):
                    self.cooccurrence_bitmaps[key] = bits.difference(expired)
                    updated += 1
            return updated
        if "format_type" in sql:
            # Requirement 06 preflight probes the bitmap payload types.
            if len(args) > 1 and args[1] in ("event_bitmap", "neighbor_bitmap"):
                return "roaringbitmap64"
            if args and str(args[0]).endswith(".neighbor_bitmaps"):
                return '"char"'
            # The fake catalog models the fully migrated VECTOR(1024) column.
            return f"vector({self.embedding_dimension})"
        if "indisprimary" in sql:
            table = args[0].rsplit(".", 1)[-1] if args else ""
            return "atom_id,anchor_id,role_type" if table == "neighbor_bitmaps" else "atom_id,anchor_id"
        if "pg_get_constraintdef" in sql:
            if "role_type" in sql:
                return "CHECK ((role_type = ANY (ARRAY['S'::\"char\", 'O'::\"char\", 'N'::\"char\"])))"
            return "CHECK ((anchor_id > 0))"
        if "column_default" in sql and "anchor_id" in sql:
            return None  # the migrated shape carries no DEFAULT 0
        if "pg_proc" in sql:
            return True  # both required function names are installed
        if "rb64_or(" in sql:
            return True  # the installed signature is callable
        if "pg_try_advisory_lock" in sql:
            return self.advisory_lock_available
        # Requirement 05A rebuild: staging and active-atom coverage counts.
        if "noesis_embedding_atom_stage" in sql and "count(*)" in sql:
            return len(self.embedding_atom_stage)
        if "noesis_embedding_anchor_stage" in sql and "count(*)" in sql:
            return len(self.embedding_anchor_stage)
        if "status = 'active'" in sql and "count(*)" in sql:
            return sum(
                1
                for (text, atom_type), atom in self.atoms.items()
                if atom_type in ("E", "P") and atom.get("status", "active") == "active"
            )
        if "embedding IS NOT NULL" in sql and "count(*)" in sql:
            return sum(
                1
                for (text, atom_type), atom in self.atoms.items()
                if atom_type in ("E", "P") and atom["embedding"] is not None
            )
        if ".anchors" in sql and "count(*)" in sql:
            # Requirement 05A §5.2: the empty-store claim must also prove zero
            # anchors before it may guess a generation.
            return len(self.anchors)
        # The ingestion preflight is read-only; the in-memory store models the
        # fully migrated schema and therefore answers every catalog probe true.
        return True

    async def execute(self, sql: str, *args: Any) -> str:
        self._maybe_fail(sql)
        self.calls.append(("execute", sql, args))
        # Requirement 06 dispatches first: the bitmap upserts carry no
        # RETURNING and are the only write path into these tables.
        if ".cooccurrence_bitmaps" in sql:
            return self._bitmap_cooccurrence_upsert(args)
        if ".neighbor_bitmaps" in sql:
            return self._bitmap_neighbor_upsert(args)
        if ".anchors" in sql and "total_count = total_count + 1" in sql:
            return self._ema_update_anchor(args)
        if ".event_atoms" in sql:
            self.event_atoms.append(args)
            return "INSERT 0 1"
        if "pg_advisory_unlock" in sql:
            return "SELECT 1"
        if sql.startswith("CREATE TEMP TABLE"):
            return "CREATE TABLE"
        if sql.startswith("TRUNCATE TABLE pg_temp.noesis_embedding_atom_stage"):
            self.embedding_atom_stage.clear()
            return "TRUNCATE TABLE"
        if sql.startswith("TRUNCATE TABLE pg_temp.noesis_embedding_context_stage"):
            self.embedding_context_stage.clear()
            return "TRUNCATE TABLE"
        if sql.startswith("TRUNCATE TABLE pg_temp.noesis_embedding_anchor_sample_stage"):
            self.embedding_anchor_sample_stage.clear()
            return "TRUNCATE TABLE"
        if sql.startswith("TRUNCATE TABLE pg_temp.noesis_embedding_anchor_stage"):
            self.embedding_anchor_stage.clear()
            return "TRUNCATE TABLE"
        if sql.startswith("DROP TABLE IF EXISTS pg_temp.noesis_embedding_atom_stage"):
            self.embedding_atom_stage.clear()
            return "DROP TABLE"
        if sql.startswith("DROP TABLE IF EXISTS pg_temp.noesis_embedding_context_stage"):
            self.embedding_context_stage.clear()
            return "DROP TABLE"
        if sql.startswith("DROP TABLE IF EXISTS pg_temp.noesis_embedding_anchor_sample_stage"):
            self.embedding_anchor_sample_stage.clear()
            return "DROP TABLE"
        if sql.startswith("DROP TABLE IF EXISTS pg_temp.noesis_embedding_anchor_stage"):
            self.embedding_anchor_stage.clear()
            return "DROP TABLE"
        if sql.startswith("INSERT INTO pg_temp.noesis_embedding_atom_stage"):
            self.embedding_atom_stage[args[0]] = _parse_vector_literal(args[1])
            return "INSERT 0 1"
        if sql.startswith("INSERT INTO pg_temp.noesis_embedding_context_stage"):
            self.embedding_context_stage[(args[0], args[1])] = _parse_vector_literal(args[2])
            return "INSERT 0 1"
        if sql.startswith("INSERT INTO pg_temp.noesis_embedding_anchor_sample_stage"):
            self.embedding_anchor_sample_stage[(args[0], args[1], args[2])] = True
            return "INSERT 0 1"
        if sql.startswith("INSERT INTO pg_temp.noesis_embedding_anchor_stage"):
            self.embedding_anchor_stage[args[0]] = (_parse_vector_literal(args[1]), int(args[2]))
            return "INSERT 0 1"
        if sql.startswith("REINDEX INDEX"):
            return "REINDEX"
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
        if ".atoms" in sql and "FROM pg_temp.noesis_embedding_atom_stage" in sql:
            null_guard = "a.embedding IS NULL" in sql
            updated = 0
            for atom in self.atoms.values():
                vector = self.embedding_atom_stage.get(atom["atom_id"])
                if vector is None or (null_guard and atom["embedding"] is not None):
                    continue
                atom["embedding"] = vector
                updated += 1
            return f"UPDATE {updated}"
        if ".anchors" in sql and "FROM pg_temp.noesis_embedding_anchor_stage" in sql:
            updated = 0
            for anchor_id, (centroid, total_count) in self.embedding_anchor_stage.items():
                anchor = self.anchors.get(anchor_id)
                if anchor is None:
                    continue
                anchor["centroid"] = tuple(centroid)
                anchor["total_count"] = int(total_count)
                updated += 1
            return f"UPDATE {updated}"
        raise AssertionError(f"unexpected execute SQL: {sql}")

    # -- table emulation ----------------------------------------------------

    def _insert_event(self, args: tuple) -> dict[str, Any] | None:
        # (event_time, data_json, source, category) — the sequence-numbered
        # INSERT ... RETURNING always yields a fresh event_id (no ON CONFLICT).
        event_time, data_json, source, category = args[0], args[1], args[2], args[3]
        self._ids["event"] += 1
        self.events[self._ids["event"]] = {
            "event_id": self._ids["event"],
            "event_time": event_time,
            "data": json.loads(data_json),
            "source": source,
            "category": category,
        }
        return {"event_id": self._ids["event"]}

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
                "embedding": vector,
            }
        else:
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

    def _lock_atom(self, args: tuple) -> dict[str, Any] | None:
        # SELECT ... FROM {s}.atoms WHERE atom_id = $1 FOR UPDATE — a missing
        # atom row returns None exactly like PostgreSQL would.
        atom_id = args[0]
        if any(atom["atom_id"] == atom_id for atom in self.atoms.values()):
            self.locked_atoms.append(atom_id)
            return {"atom_id": atom_id}
        return None

    def _insert_anchor(self, args: tuple) -> dict[str, Any]:
        # INSERT ... VALUES ($1, $2::vector, 1, 'A') RETURNING anchor_id — the
        # frozen SQL carries the explicit total_count=1 and status 'A'.
        atom_id, vector_literal = args[0], args[1]
        centroid = _parse_vector_literal(vector_literal)
        if centroid is None or len(centroid) != self.embedding_dimension:
            raise InjectedFailure("pgvector dimension reject")
        self._ids["anchor"] += 1
        anchor_id = self._ids["anchor"]
        self.anchors[anchor_id] = {
            "anchor_id": anchor_id,
            "atom_id": atom_id,
            "centroid": tuple(centroid),
            "total_count": 1,
            "status": "A",
        }
        return {"anchor_id": anchor_id}

    def _nearest_anchor(self, args: tuple) -> dict[str, Any] | None:
        # Nearest active anchor of the atom by pgvector cosine distance, with
        # the frozen (distance, anchor_id) ascending tie-break.
        atom_id, vector_literal = args[0], args[1]
        query = _parse_vector_literal(vector_literal)
        if query is None:
            raise InjectedFailure("pgvector dimension reject")
        best: tuple[int, float, tuple[float, ...]] | None = None
        for anchor_id, anchor in self.anchors.items():
            if anchor["atom_id"] != atom_id or anchor["status"] != "A":
                continue
            candidate = (anchor_id, _cosine_distance(query, anchor["centroid"]), anchor["centroid"])
            if best is None or (candidate[1], candidate[0]) < (best[1], best[0]):
                best = candidate
        if best is None:
            return None
        anchor_id, distance, centroid = best
        return {
            "anchor_id": anchor_id,
            "centroid_text": _format_vector_literal(centroid),
            "distance": distance,
        }

    def _ema_update_anchor(self, args: tuple) -> str:
        # UPDATE ... SET centroid_vector = $2::vector, total_count = total_count + 1
        anchor_id, vector_literal = args[0], args[1]
        centroid = _parse_vector_literal(vector_literal)
        if centroid is None or len(centroid) != self.embedding_dimension:
            raise InjectedFailure("pgvector dimension reject")
        anchor = self.anchors[anchor_id]
        anchor["centroid"] = tuple(centroid)
        anchor["total_count"] += 1
        return "UPDATE 1"

    def _bitmap_cooccurrence_upsert(self, args: tuple) -> str:
        # INSERT ... VALUES ($1, $2, rb64_build(ARRAY[$3]::bigint[]))
        # ON CONFLICT (atom_id, anchor_id) DO UPDATE SET event_bitmap = rb64_or(...)
        atom_id, anchor_id, event_id = int(args[0]), int(args[1]), int(args[2])
        self.cooccurrence_bitmaps.setdefault((atom_id, anchor_id), set()).add(event_id)
        return "INSERT 0 1"

    def _bitmap_neighbor_upsert(self, args: tuple) -> str:
        # INSERT ... VALUES ($1, $2, $3::text::"char", rb64_build($4::bigint[]))
        # ON CONFLICT (atom_id, anchor_id, role_type) DO UPDATE SET neighbor_bitmap = rb64_or(...)
        atom_id, anchor_id, role_type, neighbor_ids = int(args[0]), int(args[1]), str(args[2]), args[3]
        bucket = self.neighbor_bitmaps.setdefault((atom_id, anchor_id, role_type), set())
        bucket.update(int(neighbor_id) for neighbor_id in neighbor_ids)
        return "INSERT 0 1"

    # -- helpers ------------------------------------------------------------

    def event_rows(self) -> list[dict[str, Any]]:
        return list(self.events.values())

    def alerts_by_code(self, code: str) -> list[dict[str, Any]]:
        return [alert for alert in self.alerts if alert["alert_code"] == code]

    def embedding_of(self, text: str, atom_type: str) -> tuple | None:
        return self.atoms.get((text, atom_type), {}).get("embedding")

    def anchors_rows(self) -> list[dict[str, Any]]:
        return list(self.anchors.values())

    def cooccurrence_bucket(self, atom_id: int, anchor_id: int) -> set[int]:
        """Committed event IDs of one (atom_id, anchor_id) cooccurrence row."""
        return set(self.cooccurrence_bitmaps.get((atom_id, anchor_id), set()))

    def neighbor_bucket(self, atom_id: int, anchor_id: int, role_type: str) -> set[int]:
        """Committed neighbor atom IDs of one (atom_id, anchor_id, role_type) row."""
        return set(self.neighbor_bitmaps.get((atom_id, anchor_id, role_type), set()))

    def seed_anchor(
        self, atom_id: int, centroid: tuple[float, ...], *, status: str = "A", total_count: int = 0
    ) -> int:
        """Insert a pre-existing anchor row and return its anchor_id."""
        self._ids["anchor"] += 1
        anchor_id = self._ids["anchor"]
        self.anchors[anchor_id] = {
            "anchor_id": anchor_id,
            "atom_id": atom_id,
            "centroid": tuple(centroid),
            "total_count": total_count,
            "status": status,
        }
        return anchor_id

    def fetch_calls(self, marker: str) -> list[tuple]:
        return [call for call in self.calls if marker in call[1]]


def _parse_vector_literal(literal: str | None) -> tuple[float, ...] | None:
    """Parse the ``$3::vector`` text literal back into floats (fake pgvector)."""
    if literal is None:
        return None
    body = literal.strip()
    assert body.startswith("[") and body.endswith("]"), f"not a vector literal: {literal!r}"
    return tuple(float(component) for component in body[1:-1].split(","))


def _format_vector_literal(vector: Any) -> str:
    """Inverse of ``_parse_vector_literal``: repr(float) keeps full precision."""
    return "[" + ",".join(repr(float(component)) for component in vector) + "]"


def _cosine_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """pgvector ``<=>`` semantics: 1 - dot/(|left|*|right|), zero norm -> 1.0, clamped to [0, 2]."""
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if norm_left == 0.0 or norm_right == 0.0:
        return 1.0
    return max(0.0, min(2.0, 1.0 - dot / (norm_left * norm_right)))


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
                self._store.anchors,
                self._store.event_atoms,
                self._store.alerts,
                self._store.profile,
                self._store.embedding_atom_stage,
                self._store.embedding_context_stage,
                self._store.embedding_anchor_sample_stage,
                self._store.embedding_anchor_stage,
                self._store.cooccurrence_bitmaps,
                self._store.neighbor_bitmaps,
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
        (
            events,
            atoms,
            anchors,
            event_atoms,
            alerts,
            profile,
            atom_stage,
            context_stage,
            sample_stage,
            anchor_stage,
            cooccurrence_bitmaps,
            neighbor_bitmaps,
            ids,
        ) = self._snapshot
        self._store.events = events
        self._store.atoms = atoms
        self._store.anchors = anchors
        self._store.event_atoms = event_atoms
        self._store.alerts = alerts
        self._store.profile = profile
        self._store.embedding_atom_stage = atom_stage
        self._store.embedding_context_stage = context_stage
        self._store.embedding_anchor_sample_stage = sample_stage
        self._store.embedding_anchor_stage = anchor_stage
        self._store.cooccurrence_bitmaps = cooccurrence_bitmaps
        self._store.neighbor_bitmaps = neighbor_bitmaps
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
# Embedding seams (requirements 03 + 05, unified client)
# ---------------------------------------------------------------------------

def fake_identity_vector(text: str, atom_type: str, dimension: int = 1024) -> tuple[float, ...]:
    """Deterministic stand-in for BGE(literal): sha256-seeded floats in [-1, 1)."""
    digest = hashlib.sha256(f"{atom_type}::{text}".encode("utf-8")).digest()
    return tuple(round((digest[index % len(digest)] / 255.0) * 2 - 1, 6) for index in range(dimension))


def fake_context_vector(context_text: str, dimension: int = 1024) -> tuple[float, ...]:
    """Deterministic stand-in for BGE(context_text): sha256-seeded floats in [-1, 1)."""
    digest = hashlib.sha256(f"context::{context_text}".encode("utf-8")).digest()
    return tuple(round((digest[index % len(digest)] / 255.0) * 2 - 1, 6) for index in range(dimension))


class FakeEmbeddingClient:
    """``embedding_client_factory`` seam: scriptable bge stand-in covering both
    the identity endpoints and the ``/normalize/sentence`` context endpoint.

    ``health`` is "ready" (default), "config_invalid", or "unavailable";
    ``identity_failures`` maps (text, atom_type) -> error_kind;
    ``context_failures`` maps context_text -> error_kind.
    """

    def __init__(
        self,
        *,
        health: str = "ready",
        identity_failures: dict[tuple[str, str], str] | None = None,
        context_failures: dict[str, str] | None = None,
        dimension: int = 1024,
    ) -> None:
        self.health = health
        self.identity_failures = dict(identity_failures or {})
        self.context_failures = dict(context_failures or {})
        self.dimension = dimension
        self.ensure_calls = 0
        self.identity_calls: list[tuple[str, str]] = []
        self.context_calls: list[str] = []
        self.closed = False

    async def ensure_ready(self) -> None:
        self.ensure_calls += 1
        if self.health == "config_invalid":
            raise EmbeddingConfigInvalid("fake /health config mismatch")
        if self.health == "unavailable":
            raise EmbeddingServiceUnavailable("fake /health transport failure")

    async def embed_identity(self, text: str, atom_type: str) -> list[float]:
        self.identity_calls.append((text, atom_type))
        error_kind = self.identity_failures.get((text, atom_type))
        if error_kind is not None:
            raise EmbeddingError(error_kind, f"fake embed failure: {error_kind}")
        return list(fake_identity_vector(text, atom_type, self.dimension))

    async def embed_context(self, context_text: str) -> list[float]:
        self.context_calls.append(context_text)
        error_kind = self.context_failures.get(context_text)
        if error_kind is not None:
            raise EmbeddingError(error_kind, f"fake context failure: {error_kind}")
        return list(fake_context_vector(context_text, self.dimension))

    async def aclose(self) -> None:
        self.closed = True


def embedding_factory_for(client: FakeEmbeddingClient):
    """``embedding_client_factory`` seam returning ``client`` for any config."""

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
