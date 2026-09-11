"""Noesis Stage 1 cooccurrence bitmap window pruning (requirement 07).

Focused lifecycle module for the ``cooccurrence_bitmaps.event_bitmap`` payload:
the lifecycle coordinator hands over a batch of already-decided expired Event
IDs, and this module removes exactly those bits in PostgreSQL with a single
database-side ``rb64_andnot`` statement. It never decides which Events are
expired, never computes a cutoff, never touches the Neighbor bitmap (an
all-history accumulator that is never pruned), and never updates, deletes, or
reads the Event tables.

Boundaries frozen by requirement 07:

* input validation is strict: positive Python ``int`` only (``bool`` rejected
  even though it subclasses ``int``), deduplicated, ascending, at most 10,000
  unique IDs per batch, whole-batch rejection on any bad value;
* an empty batch returns zero evidence without touching the connection;
* exactly one statement per batch builds the expired bitmap once in SQL
  (``rb64_build($1::bigint[])``), updates only buckets where
  ``rb64_and_cardinality(event_bitmap, expired) > 0``, and returns the number
  of touched buckets from ``RETURNING``; a bucket emptied by the prune keeps
  its row with an empty bitmap;
* the caller owns the connection and the transaction: no pool acquire, no
  nested transaction, no commit or rollback;
* database errors are wrapped in :class:`CooccurrenceRetentionError` with the
  original ``__cause__`` preserved; :class:`asyncio.CancelledError` is a
  ``BaseException`` and propagates untouched;
* the read-only preflight validates the table payload type, the strict
  ``(atom_id, anchor_id)`` primary key, and that ``rb64_build`` /
  ``rb64_andnot`` / ``rb64_and_cardinality`` are installed and callable. The
  retention lifecycle may run without any fact ingestion having happened, so
  this preflight never relies on another subsystem's checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ...config import noesis_schema_is_valid

MAX_EXPIRED_EVENT_IDS_PER_BATCH = 10_000
_BIGINT_MAX = 2**63 - 1

_PRUNE_SQL = (
    "WITH expired AS ("
    "SELECT rb64_build($1::bigint[]) AS event_bitmap"
    "), updated AS ("
    "UPDATE {s}.cooccurrence_bitmaps AS c "
    "SET event_bitmap = rb64_andnot(c.event_bitmap, expired.event_bitmap) "
    "FROM expired "
    "WHERE rb64_and_cardinality(c.event_bitmap, expired.event_bitmap) > 0 "
    "RETURNING c.atom_id, c.anchor_id"
    ") SELECT count(*)::bigint FROM updated"
)


class CooccurrenceRetentionInputError(ValueError):
    """The caller passed an invalid schema or an invalid expired-event batch."""


class CooccurrenceRetentionError(Exception):
    """The cooccurrence prune or its preflight failed inside the database."""


@dataclass(frozen=True)
class ExpiredEventBatch:
    """Validated, deduplicated, ascending expired Event IDs."""

    event_ids: tuple[int, ...]


@dataclass(frozen=True)
class CooccurrencePruneResult:
    """Evidence for one batch: requested, unique, and actually updated buckets."""

    requested_ids: int
    unique_ids: int
    updated_buckets: int


def _sql(schema: str, statement: str) -> str:
    return statement.format(s=schema)


def plan_expired_event_batch(event_ids: Iterable[int]) -> ExpiredEventBatch:
    """Validate one expired batch: positive BIGINT ints, deduplicated, ascending.

    The input is materialized exactly once. Any invalid element fails the whole
    batch; nothing is converted, skipped, or truncated, and more than
    ``MAX_EXPIRED_EVENT_IDS_PER_BATCH`` unique IDs is rejected outright.
    """
    raw = tuple(event_ids)
    for event_id in raw:
        if isinstance(event_id, bool) or not isinstance(event_id, int):
            raise CooccurrenceRetentionInputError("expired event IDs must be integers")
        if event_id <= 0 or event_id > _BIGINT_MAX:
            raise CooccurrenceRetentionInputError("expired event ID is outside positive BIGINT range")
    unique = tuple(sorted(set(raw)))
    if len(unique) > MAX_EXPIRED_EVENT_IDS_PER_BATCH:
        raise CooccurrenceRetentionInputError("expired event-ID batch exceeds 10000 unique IDs")
    return ExpiredEventBatch(unique)


async def check_cooccurrence_retention_ready(conn: Any, schema: str) -> None:
    """Read-only preflight for the cooccurrence retention path.

    Explicitly invoked once per sweep by the lifecycle coordinator (never
    repeated before every batch): validates the schema identifier, the
    ``roaringbitmap64`` payload type, the strict ``(atom_id, anchor_id)``
    primary key, and that the required ``rb64_*`` functions are installed and
    callable.
    """
    if not noesis_schema_is_valid(schema):
        raise CooccurrenceRetentionInputError("invalid Noesis schema identifier")
    declared = await conn.fetchval(
        "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
        "WHERE a.attrelid = to_regclass($1) AND a.attname = 'event_bitmap'",
        f"{schema}.cooccurrence_bitmaps",
    )
    if declared != "roaringbitmap64":
        raise CooccurrenceRetentionError(
            f"'{schema}.cooccurrence_bitmaps.event_bitmap' declares {declared!r}, expected 'roaringbitmap64'"
        )
    pk_columns = await conn.fetchval(
        "SELECT string_agg(a.attname, ',' ORDER BY array_position(i.indkey::int2[], a.attnum)) "
        "FROM pg_index i JOIN pg_attribute a "
        "ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
        "WHERE i.indrelid = to_regclass($1) AND i.indisprimary",
        f"{schema}.cooccurrence_bitmaps",
    )
    if pk_columns != "atom_id,anchor_id":
        raise CooccurrenceRetentionError(
            f"'{schema}.cooccurrence_bitmaps' primary key is {pk_columns!r}, expected 'atom_id,anchor_id'"
        )
    functions_present = await conn.fetchval(
        "SELECT to_regprocedure('rb64_build(bigint[])') IS NOT NULL "
        "AND to_regprocedure('rb64_andnot(roaringbitmap64,roaringbitmap64)') IS NOT NULL "
        "AND to_regprocedure('rb64_and_cardinality(roaringbitmap64,roaringbitmap64)') IS NOT NULL"
    )
    if functions_present is not True:
        raise CooccurrenceRetentionError("required roaringbitmap retention functions are not installed")
    try:
        callable_result = await conn.fetchval(
            "SELECT rb64_and_cardinality("
            "rb64_andnot(rb64_build(ARRAY[1::bigint]), rb64_build(ARRAY[1::bigint])), "
            "rb64_build(ARRAY[2::bigint])) = 0"
        )
    except Exception as error:
        raise CooccurrenceRetentionError(
            f"roaringbitmap retention functions are not callable: {type(error).__name__}"
        ) from error
    if callable_result is not True:
        raise CooccurrenceRetentionError("roaringbitmap retention function probe returned an invalid result")


async def prune_cooccurrence_event_ids(
    conn: Any, schema: str, event_ids: Iterable[int]
) -> CooccurrencePruneResult:
    """Remove the expired Event bits from ``cooccurrence_bitmaps`` in one batch.

    Runs on the caller's open transaction connection: no pool acquire, no
    nested transaction, no commit/rollback, no swallowed exception. The caller
    supplies already-decided expired Event IDs; this function never queries or
    judges Event state, never reads or writes the Event tables, and never
    touches the Neighbor bitmap.
    """
    if not noesis_schema_is_valid(schema):
        raise CooccurrenceRetentionInputError("invalid Noesis schema identifier")
    raw = tuple(event_ids)
    batch = plan_expired_event_batch(raw)
    if not batch.event_ids:
        return CooccurrencePruneResult(len(raw), 0, 0)
    try:
        updated = await conn.fetchval(
            _sql(schema, _PRUNE_SQL),
            list(batch.event_ids),
        )
    except Exception as error:
        raise CooccurrenceRetentionError(
            f"cooccurrence retention failed: {type(error).__name__}"
        ) from error
    return CooccurrencePruneResult(len(raw), len(batch.event_ids), int(updated))
