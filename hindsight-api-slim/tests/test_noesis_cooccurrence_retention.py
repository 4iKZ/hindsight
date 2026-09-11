"""Requirement 07 cooccurrence retention tests (input planning / ANDNOT / transaction).

Pure input-planning units (requirement 07 §11.1), the read-only retention
preflight (table type / strict primary key / rb64 functions), the database-side
ANDNOT behavior against the in-memory FakeStore (§11.2), Neighbor invariants
(§11.3), SQL shape (§11.4), transaction/error semantics (§11.5), and concurrent
cleanup / OR-ANDNOT interleaving (§11.6). No network, no real database, no
LLM/BGE.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import hindsight_api.engine.retain.noesis_cooccurrence_retention as retention
from hindsight_api.engine.retain.noesis_bitmap import (
    BitmapWritePlan,
    CooccurrenceBitmapWrite,
    apply_bitmap_writes,
)
from hindsight_api.engine.retain.noesis_cooccurrence_retention import (
    MAX_EXPIRED_EVENT_IDS_PER_BATCH,
    CooccurrencePruneResult,
    CooccurrenceRetentionError,
    CooccurrenceRetentionInputError,
    ExpiredEventBatch,
    check_cooccurrence_retention_ready,
    plan_expired_event_batch,
    prune_cooccurrence_event_ids,
)
from tests.noesis_fakes import FakeConn, FakeStore

SCHEMA = "noesis_core"
BIGINT_MAX = 2**63 - 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def seed_buckets(store: FakeStore, buckets: dict[tuple[int, int], set[int]]) -> None:
    for key, bits in buckets.items():
        store.cooccurrence_bitmaps[key] = set(bits)


def snapshot(store: FakeStore) -> dict:
    return {
        "cooccurrence": {key: set(bits) for key, bits in store.cooccurrence_bitmaps.items()},
        "neighbor": {key: set(bits) for key, bits in store.neighbor_bitmaps.items()},
        "tx_log": list(store.tx_log),
        "commits": store.commits,
        "rollbacks": store.rollbacks,
    }


class _BoomConn:
    """Every SQL call fails with the given error."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[str] = []

    async def fetchval(self, sql, *args):
        self.calls.append(sql)
        raise self.error


class _CancelConn:
    async def fetchval(self, sql, *args):
        raise asyncio.CancelledError()


class _StrictConn:
    """Only ``fetchval`` exists: any transaction/execute/acquire call would AttributeError."""

    def __init__(self, store: FakeStore) -> None:
        self._store = store

    async def fetchval(self, sql, *args):
        return await self._store.fetchval(sql, *args)


class _PreflightConn:
    """Scripted catalog answers for the retention preflight probes."""

    def __init__(
        self,
        *,
        event_bitmap_type: str | None = "roaringbitmap64",
        pk: str | None = "atom_id,anchor_id",
        functions_present: bool = True,
        callable_result: bool = True,
        callable_error: Exception | None = None,
    ) -> None:
        self.event_bitmap_type = event_bitmap_type
        self.pk = pk
        self.functions_present = functions_present
        self.callable_result = callable_result
        self.callable_error = callable_error
        self.queries: list[str] = []

    async def fetchval(self, sql, *args):
        self.queries.append(sql)
        if "format_type" in sql:
            return self.event_bitmap_type
        if "indisprimary" in sql:
            return self.pk
        if "to_regprocedure" in sql:
            return self.functions_present
        if "rb64_andnot(" in sql:
            if self.callable_error is not None:
                raise self.callable_error
            return self.callable_result
        raise AssertionError(f"unexpected preflight SQL: {sql}")


# ---------------------------------------------------------------------------
# §11.1 input planning
# ---------------------------------------------------------------------------


def test_plan_deduplicates_and_sorts_event_ids():
    batch = plan_expired_event_batch([9, 3, 9, 5])
    assert batch.event_ids == (3, 5, 9)


@pytest.mark.parametrize("bad", [0, -1, True, "12", 1.5, None, 2**63])
def test_plan_rejects_invalid_event_id(bad):
    with pytest.raises(CooccurrenceRetentionInputError):
        plan_expired_event_batch([1, bad])


def test_plan_accepts_bigint_max():
    assert plan_expired_event_batch([BIGINT_MAX]).event_ids == (BIGINT_MAX,)


def test_plan_accepts_10000_unique_ids():
    ids = list(range(1, MAX_EXPIRED_EVENT_IDS_PER_BATCH + 1))
    assert plan_expired_event_batch(ids).event_ids == tuple(ids)


def test_plan_rejects_10001_unique_ids_without_truncation():
    ids = range(1, MAX_EXPIRED_EVENT_IDS_PER_BATCH + 2)
    with pytest.raises(CooccurrenceRetentionInputError):
        plan_expired_event_batch(ids)


def test_plan_accepts_10001_inputs_deduplicated_below_limit():
    assert plan_expired_event_batch([7] * 10_001).event_ids == (7,)


def test_plan_consumes_generator_once():
    consumed = 0

    def generator():
        nonlocal consumed
        for value in (5, 3, 5):
            consumed += 1
            yield value

    batch = plan_expired_event_batch(generator())
    assert batch.event_ids == (3, 5)
    assert consumed == 3


def test_plan_fails_whole_batch_on_late_bad_value():
    with pytest.raises(CooccurrenceRetentionInputError):
        plan_expired_event_batch([1, 2, 3, "4"])


def test_plan_empty_returns_empty_batch():
    assert plan_expired_event_batch([]) == ExpiredEventBatch(())


# ---------------------------------------------------------------------------
# read-only retention preflight: type, strict PK, rb64 functions
# ---------------------------------------------------------------------------


async def test_preflight_passes_on_correct_catalog():
    conn = _PreflightConn()
    await check_cooccurrence_retention_ready(conn, SCHEMA)
    assert len(conn.queries) == 4


async def test_preflight_rejects_invalid_schema_before_sql():
    conn = _PreflightConn()
    with pytest.raises(CooccurrenceRetentionInputError):
        await check_cooccurrence_retention_ready(conn, "bad; DROP SCHEMA")
    assert conn.queries == []


async def test_preflight_rejects_wrong_event_bitmap_type():
    with pytest.raises(CooccurrenceRetentionError, match="roaringbitmap64"):
        await check_cooccurrence_retention_ready(_PreflightConn(event_bitmap_type="roaringbitmap32"), SCHEMA)


async def test_preflight_rejects_missing_event_bitmap_column():
    with pytest.raises(CooccurrenceRetentionError, match="roaringbitmap64"):
        await check_cooccurrence_retention_ready(_PreflightConn(event_bitmap_type=None), SCHEMA)


@pytest.mark.parametrize("pk", [None, "", "atom_id", "anchor_id,atom_id", "atom_id,anchor_id,role_type"])
async def test_preflight_rejects_wrong_primary_key(pk):
    with pytest.raises(CooccurrenceRetentionError, match="primary key"):
        await check_cooccurrence_retention_ready(_PreflightConn(pk=pk), SCHEMA)


async def test_preflight_rejects_missing_functions():
    with pytest.raises(CooccurrenceRetentionError, match="not installed"):
        await check_cooccurrence_retention_ready(_PreflightConn(functions_present=False), SCHEMA)


async def test_preflight_wraps_uncallable_functions_and_keeps_cause():
    with pytest.raises(CooccurrenceRetentionError) as excinfo:
        await check_cooccurrence_retention_ready(_PreflightConn(callable_error=RuntimeError("boom")), SCHEMA)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


async def test_preflight_rejects_invalid_callable_probe_result():
    with pytest.raises(CooccurrenceRetentionError, match="invalid result"):
        await check_cooccurrence_retention_ready(_PreflightConn(callable_result=False), SCHEMA)


# ---------------------------------------------------------------------------
# §11.2 ANDNOT correctness against the fake store
# ---------------------------------------------------------------------------


async def test_prune_removes_only_expired_bits():
    store = FakeStore()
    seed_buckets(store, {(1, 10): {1, 2, 3, 4}, (2, 10): {2, 5}, (3, 10): {6, 7}})

    result = await prune_cooccurrence_event_ids(FakeConn(store), SCHEMA, [2, 3, 99])

    assert result == CooccurrencePruneResult(requested_ids=3, unique_ids=3, updated_buckets=2)
    assert store.cooccurrence_bucket(1, 10) == {1, 4}
    assert store.cooccurrence_bucket(2, 10) == {5}
    assert store.cooccurrence_bucket(3, 10) == {6, 7}


async def test_prune_reports_requested_unique_and_updated_counts():
    store = FakeStore()
    for index, bit in enumerate([3, 5, 9, 3, 5, 9, 3], start=1):
        store.cooccurrence_bitmaps[(index, 10)] = {bit}

    result = await prune_cooccurrence_event_ids(FakeConn(store), SCHEMA, [9, 3, 9, 5])

    assert result.requested_ids == 4
    assert result.unique_ids == 3
    assert result.updated_buckets == 7


async def test_prune_missing_ids_returns_zero_updates():
    store = FakeStore()
    seed_buckets(store, {(1, 10): {1, 2}})

    result = await prune_cooccurrence_event_ids(FakeConn(store), SCHEMA, [998, 999])

    assert result.updated_buckets == 0
    assert store.cooccurrence_bucket(1, 10) == {1, 2}


async def test_prune_repeat_is_idempotent():
    store = FakeStore()
    seed_buckets(store, {(1, 10): {1, 2, 3}})
    conn = FakeConn(store)

    first = await prune_cooccurrence_event_ids(conn, SCHEMA, [2, 3])
    second = await prune_cooccurrence_event_ids(conn, SCHEMA, [2, 3])

    assert first.updated_buckets == 1
    assert second.updated_buckets == 0
    assert store.cooccurrence_bucket(1, 10) == {1}


async def test_prune_keeps_fully_emptied_row():
    store = FakeStore()
    seed_buckets(store, {(1, 10): {5}})

    result = await prune_cooccurrence_event_ids(FakeConn(store), SCHEMA, [5])

    assert result.updated_buckets == 1
    assert (1, 10) in store.cooccurrence_bitmaps, "empty bucket rows must be kept, not deleted"
    assert store.cooccurrence_bitmaps[(1, 10)] == set()


async def test_prune_duplicate_input_matches_unique_result():
    store = FakeStore()
    seed_buckets(store, {(1, 10): {2, 4}})

    result = await prune_cooccurrence_event_ids(FakeConn(store), SCHEMA, [2, 2, 2])

    assert result.requested_ids == 3
    assert result.unique_ids == 1
    assert store.cooccurrence_bucket(1, 10) == {4}


async def test_prune_handles_bigint_max_member():
    store = FakeStore()
    seed_buckets(store, {(1, 10): {1, BIGINT_MAX}})

    result = await prune_cooccurrence_event_ids(FakeConn(store), SCHEMA, [BIGINT_MAX])

    assert result.updated_buckets == 1
    assert store.cooccurrence_bucket(1, 10) == {1}


async def test_prune_empty_input_issues_zero_sql():
    store = FakeStore()

    result = await prune_cooccurrence_event_ids(FakeConn(store), SCHEMA, [])

    assert result == CooccurrencePruneResult(0, 0, 0)
    assert store.calls == [], "empty input must not touch the database"


async def test_prune_invalid_schema_rejected_before_sql():
    store = FakeStore()

    with pytest.raises(CooccurrenceRetentionInputError):
        await prune_cooccurrence_event_ids(FakeConn(store), "bad; DROP SCHEMA", [1])

    assert store.calls == []


async def test_prune_invalid_event_ids_rejected_before_sql():
    store = FakeStore()
    seed_buckets(store, {(1, 10): {1}})

    with pytest.raises(CooccurrenceRetentionInputError):
        await prune_cooccurrence_event_ids(FakeConn(store), SCHEMA, [1, 0])

    assert store.calls == []
    assert store.cooccurrence_bucket(1, 10) == {1}


# ---------------------------------------------------------------------------
# §11.3 Neighbor invariants
# ---------------------------------------------------------------------------


async def test_neighbor_bitmaps_unchanged_on_every_path():
    store = FakeStore()
    store.neighbor_bitmaps[(1, 10, "N")] = {42, 43}
    seed_buckets(store, {(1, 10): {1, 2, 3}})
    before = {key: set(bits) for key, bits in store.neighbor_bitmaps.items()}
    conn = FakeConn(store)

    await prune_cooccurrence_event_ids(conn, SCHEMA, [])
    await prune_cooccurrence_event_ids(conn, SCHEMA, [1])
    await prune_cooccurrence_event_ids(conn, SCHEMA, [1])
    with pytest.raises(CooccurrenceRetentionInputError):
        await prune_cooccurrence_event_ids(conn, SCHEMA, [0])
    store.fail_after = {"rb64_andnot": 0}
    with pytest.raises(CooccurrenceRetentionError):
        async with conn.transaction():
            await prune_cooccurrence_event_ids(conn, SCHEMA, [2])
    store.fail_after = {}
    with pytest.raises(RuntimeError):
        async with conn.transaction():
            await prune_cooccurrence_event_ids(conn, SCHEMA, [3])
            raise RuntimeError("caller aborts")

    assert store.neighbor_bitmaps == before, "Neighbor bitmaps are all-history and must never change"
    assert not any(".neighbor_bitmaps" in sql for _kind, sql, _args in store.calls)


# ---------------------------------------------------------------------------
# §11.4 SQL shape
# ---------------------------------------------------------------------------


async def test_prune_sql_shape_and_forbidden_tokens():
    store = FakeStore()
    seed_buckets(store, {(1, 10): {1}})

    await prune_cooccurrence_event_ids(FakeConn(store), SCHEMA, [1])

    sqls = [sql for kind, sql, _args in store.calls if kind == "fetchval"]
    assert len(sqls) == 1, "one single ANDNOT statement per batch"
    sql = sqls[0]
    for required in (
        "WITH expired",
        "rb64_build($1::bigint[])",
        f"UPDATE {SCHEMA}.cooccurrence_bitmaps",
        "rb64_andnot(",
        "rb64_and_cardinality(",
        "RETURNING",
    ):
        assert required in sql, f"missing SQL fragment: {required}"
    assert sql.count("rb64_build") == 1, "the expired bitmap must be constructed exactly once in SQL"
    for forbidden in (
        "rb64_to_array",
        "rb_to_array",
        "DELETE FROM",
        ".neighbor_bitmaps",
        "FROM events",
        "FROM event_atoms",
        "COMMIT",
        "ROLLBACK",
        "updated_at",
    ):
        assert forbidden not in sql, f"forbidden SQL fragment: {forbidden}"


def test_production_module_has_no_neighbor_or_event_sql():
    source = Path(retention.__file__).read_text(encoding="utf-8")
    for forbidden in ("neighbor_bitmaps", "FROM events", "event_atoms", "rb64_to_array", "DELETE FROM"):
        assert forbidden not in source, f"production module leaked: {forbidden}"


# ---------------------------------------------------------------------------
# §11.5 transaction / error semantics
# ---------------------------------------------------------------------------


async def test_prune_runs_inside_caller_transaction():
    store = FakeStore()
    seed_buckets(store, {(1, 10): {1, 2}})
    conn = FakeConn(store)

    async with conn.transaction():
        result = await prune_cooccurrence_event_ids(conn, SCHEMA, [1])

    assert result.updated_buckets == 1
    assert store.tx_log == ["begin", "commit"]
    assert store.commits == 1
    assert store.rollbacks == 0


async def test_prune_never_acquires_or_manages_transactions():
    store = FakeStore()
    seed_buckets(store, {(1, 10): {1}})
    conn = _StrictConn(store)

    result = await prune_cooccurrence_event_ids(conn, SCHEMA, [1])

    assert result.updated_buckets == 1
    assert store.tx_log == []
    assert store.commits == 0
    assert len(store.calls) == 1


async def test_prune_wraps_db_error_and_keeps_cause():
    error = RuntimeError("boom")
    with pytest.raises(CooccurrenceRetentionError) as excinfo:
        await prune_cooccurrence_event_ids(_BoomConn(error), SCHEMA, [111, 222, 333])
    assert excinfo.value.__cause__ is error
    assert str(excinfo.value) == "cooccurrence retention failed: RuntimeError"
    assert "111" not in str(excinfo.value)
    assert "222" not in str(excinfo.value)
    assert "333" not in str(excinfo.value)


async def test_prune_cancellation_propagates_untouched():
    with pytest.raises(asyncio.CancelledError):
        await prune_cooccurrence_event_ids(_CancelConn(), SCHEMA, [1])


async def test_prune_rollback_restores_all_buckets():
    store = FakeStore()
    seed_buckets(store, {(1, 10): {1, 2, 3}, (2, 10): {3}})
    before = snapshot(store)
    conn = FakeConn(store)

    with pytest.raises(RuntimeError):
        async with conn.transaction():
            result = await prune_cooccurrence_event_ids(conn, SCHEMA, [1, 3])
            assert result.updated_buckets == 2
            raise RuntimeError("caller aborts")

    assert {key: set(bits) for key, bits in store.cooccurrence_bitmaps.items()} == before["cooccurrence"]
    assert store.rollbacks == 1


async def test_prune_fake_failure_injection_is_wrapped_and_rolled_back():
    store = FakeStore()
    seed_buckets(store, {(1, 10): {1, 2}})
    before = {key: set(bits) for key, bits in store.cooccurrence_bitmaps.items()}
    store.fail_after = {"rb64_andnot": 0}
    conn = FakeConn(store)

    with pytest.raises(CooccurrenceRetentionError):
        async with conn.transaction():
            await prune_cooccurrence_event_ids(conn, SCHEMA, [1])

    assert {key: set(bits) for key, bits in store.cooccurrence_bitmaps.items()} == before
    assert store.rollbacks == 1


# ---------------------------------------------------------------------------
# §11.6 concurrency: overlapping prune batches and OR/ANDNOT interleaving
# ---------------------------------------------------------------------------


async def test_concurrent_prune_overlapping_batches_converge():
    store = FakeStore()
    store.cooccurrence_bitmaps[(1, 10)] = {1, 2, 3, 4, 5}

    async def yield_once():
        await asyncio.sleep(0)

    store.interleave = yield_once
    conn = FakeConn(store)

    async def run(ids):
        async with conn.transaction():
            return await prune_cooccurrence_event_ids(conn, SCHEMA, ids)

    await asyncio.gather(run([1, 2, 3]), run([3, 4]))

    assert store.cooccurrence_bitmaps[(1, 10)] == {5}


async def test_interleaved_or_and_prune_keep_the_new_bit():
    async def yield_once():
        await asyncio.sleep(0)

    async def scenario(or_first: bool) -> set[int]:
        store = FakeStore()
        store.cooccurrence_bitmaps[(1, 10)] = {1, 2}
        store.interleave = yield_once
        conn = FakeConn(store)
        plan = BitmapWritePlan(cooccurrences=(CooccurrenceBitmapWrite(1, 10, 3),), neighbors=())

        async def or_write():
            await apply_bitmap_writes(conn, SCHEMA, plan)

        async def prune_write():
            await prune_cooccurrence_event_ids(conn, SCHEMA, [1])

        if or_first:
            await or_write()
            await prune_write()
        else:
            await prune_write()
            await or_write()
        return store.cooccurrence_bitmaps[(1, 10)]

    assert await scenario(or_first=True) == {2, 3}
    assert await scenario(or_first=False) == {2, 3}
