"""Noesis shared embedding-space rebuild command (requirement 05A §6).

One explicit, fail-closed, low-frequency operator path that rebuilds the whole
E/P shared embedding space toward the configured generation:

* ``atoms.embedding``          — Identity Vector per active E/P atom;
* ``anchors.centroid_vector``  — predicate-frame Context EMA replay;
* ``anchors.total_count``      — the rebuilt sample count;
* the frozen IVFFlat ANN index over ``atoms.embedding`` (``REINDEX INDEX``);
* ``embedding_profiles.identity`` — flipped to the target triple + ``ready``.

The Anchor samples are reconstructed from the authoritative ``events.data``
component JSON and the existing ``event_atoms.anchor_id`` — Anchor ids and
occurrence ownership are preserved, nothing is re-routed, merged, or split
(§4.3/§6.9). Context Vectors are never persisted; ``events.context_embedding``
stays NULL.

Run modes:

* default            — full rebuild; a no-op when the profile is already
  ``ready`` and exactly equals the target generation.
* ``--force-full``   — full rebuild even when the profile already equals the
  target; used once at 05A first deployment to establish a trusted shared
  generation baseline. It never bypasses health, lock, coverage, or rollback.
* ``--repair-null-only`` — only fill NULL E/P ``atoms.embedding`` under the
  current matching ``ready`` generation; it never touches anchors, the ANN
  index, or the profile, and never scans events. Mutually exclusive with
  ``--force-full``.

Invariants (§6.3/§6.5/§6.13, §7):

* a PostgreSQL session advisory lock serializes rebuilds; a second instance
  exits non-zero fast instead of waiting or taking over;
* encoding never mutates live vectors or centroids: any failure leaves live
  data at the pre-switch version and the profile at ``rebuilding``;
* staging uses bounded DB TEMP tables and 256-row keyset pagination;
* one final short maintenance transaction switches atoms + anchors + ANN +
  profile together, or rolls the whole generation back;
* every failure is a stable non-zero exit code; the profile is never silently
  returned to ``ready`` on failure.

The CLI entry point builds a real client (``NoesisEmbeddingClient``) and a real
asyncpg pool from the Hindsight config; the async core is injected with a
client and a pool so tests can run fully offline against fake seams.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import asyncpg

from .noesis_anchor import AnchorFrameError, ema_centroid, parse_centroid_text, plan_anchor_routing, vector_literal

logger = logging.getLogger(__name__)

# Session advisory lock key (arbitrary but stable) used to serialize rebuilds.
_ADVISORY_LOCK_KEY = 0x4E4F4553_495352  # "NOESISR"

# Requirement 05A §6.12: the single frozen requirement-04 ANN index name.
_FROZEN_ANN_INDEX = "idx_atoms_embedding_ivfflat"

# Requirement 05A §6.7: fixed keyset page size.
_PAGE_SIZE = 256

# Requirement 05A §4.4/§6.11: cap the reported orphan/late rows.
_REPORT_LIMIT = 50

# Exit codes (requirement 05A §7).
_EXIT_OK = 0
_EXIT_LOCK_HELD = 2
_EXIT_ANN_UNKNOWN = 3
_EXIT_CLIENT_NOT_READY = 4
_EXIT_PROFILE_INVALID = 5
_EXIT_ENCODE_FAILED = 6
_EXIT_HISTORY_INVALID = 7
_EXIT_COVERAGE_INCOMPLETE = 8
_EXIT_CUTOVER_FAILED = 9


@dataclass(frozen=True)
class BuildSpec:
    """The target (model, revision, dimension) generation to rebuild toward."""

    model: str
    revision: str
    dimension: int


# ---------------------------------------------------------------------------
# SQL builders (requirement 05A §6.6/§6.7/§6.12/§6.13)
# ---------------------------------------------------------------------------

def _sql(schema: str, statement: str) -> str:
    return statement.format(s=schema, page_size=_PAGE_SIZE, limit=_REPORT_LIMIT)


def _profile_select(schema: str) -> str:
    return _sql(
        schema,
        "SELECT model_name, model_revision, dimension, status FROM {s}.embedding_profiles "
        "WHERE embedding_kind = 'identity' FOR UPDATE",
    )


def _profile_set_rebuilding(schema: str) -> str:
    return _sql(
        schema,
        "UPDATE {s}.embedding_profiles SET status = 'rebuilding', updated_at = now() "
        "WHERE embedding_kind = 'identity'",
    )


def _profile_set_ready(schema: str) -> str:
    return _sql(
        schema,
        "UPDATE {s}.embedding_profiles SET model_name = $1, model_revision = $2, "
        "dimension = $3, status = 'ready', updated_at = now() WHERE embedding_kind = 'identity'",
    )


def _atoms_page(schema: str, null_only: bool) -> str:
    guard = "AND embedding IS NULL " if null_only else ""
    return _sql(
        schema,
        "SELECT atom_id, text, atom_type FROM {s}.atoms "
        "WHERE status = 'active' AND atom_type IN ('E', 'P') " + guard
        + "AND atom_id > $1 ORDER BY atom_id LIMIT {page_size}",
    )


def _events_page(schema: str) -> str:
    return _sql(
        schema,
        "SELECT event_id, data FROM {s}.events WHERE event_id > $1 "
        "ORDER BY event_id LIMIT {page_size}",
    )


def _event_atoms_for_event(schema: str) -> str:
    return _sql(
        schema,
        "SELECT occurrence_id, atom_id, anchor_id FROM {s}.event_atoms "
        "WHERE event_id = $1 ORDER BY occurrence_id",
    )


def _atoms_by_ids(schema: str) -> str:
    return _sql(schema, "SELECT atom_id, text, atom_type FROM {s}.atoms WHERE atom_id = ANY($1::bigint[])")


def _anchors_by_ids(schema: str) -> str:
    return _sql(schema, "SELECT anchor_id, atom_id FROM {s}.anchors WHERE anchor_id = ANY($1::bigint[])")


def _active_ep_count(schema: str) -> str:
    return _sql(schema, "SELECT count(*) FROM {s}.atoms WHERE status = 'active' AND atom_type IN ('E', 'P')")


def _all_anchor_count(schema: str) -> str:
    return _sql(schema, "SELECT count(*) FROM {s}.anchors")


def _atom_stage_count(_schema: str) -> str:
    return "SELECT count(*) FROM pg_temp.noesis_embedding_atom_stage"


def _anchor_stage_count(_schema: str) -> str:
    return "SELECT count(*) FROM pg_temp.noesis_embedding_anchor_stage"


def _orphan_anchors(schema: str) -> str:
    return _sql(
        schema,
        "SELECT a.anchor_id, a.atom_id FROM {s}.anchors a "
        "LEFT JOIN pg_temp.noesis_embedding_anchor_stage st ON st.anchor_id = a.anchor_id "
        "WHERE st.anchor_id IS NULL ORDER BY a.anchor_id LIMIT {limit}",
    )


def _ann_index_check(schema: str) -> str:
    return _sql(
        schema,
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = $1 AND tablename = 'atoms'",
    )


_ATOM_STAGE_CREATE = (
    "CREATE TEMP TABLE IF NOT EXISTS noesis_embedding_atom_stage ("
    "atom_id BIGINT PRIMARY KEY, embedding VECTOR({dimension}) NOT NULL"
    ") ON COMMIT PRESERVE ROWS"
)
_CONTEXT_STAGE_CREATE = (
    "CREATE TEMP TABLE IF NOT EXISTS noesis_embedding_context_stage ("
    "event_id BIGINT NOT NULL, predicate_pos INT NOT NULL, "
    "context_vector VECTOR({dimension}) NOT NULL, PRIMARY KEY (event_id, predicate_pos)"
    ") ON COMMIT PRESERVE ROWS"
)
_SAMPLE_STAGE_CREATE = (
    "CREATE TEMP TABLE IF NOT EXISTS noesis_embedding_anchor_sample_stage ("
    "anchor_id BIGINT NOT NULL, event_id BIGINT NOT NULL, predicate_pos INT NOT NULL, "
    "PRIMARY KEY (anchor_id, event_id, predicate_pos)"
    ") ON COMMIT PRESERVE ROWS"
)
_ANCHOR_STAGE_CREATE = (
    "CREATE TEMP TABLE IF NOT EXISTS noesis_embedding_anchor_stage ("
    "anchor_id BIGINT PRIMARY KEY, centroid_vector VECTOR({dimension}) NOT NULL, "
    "total_count BIGINT NOT NULL CHECK (total_count > 0)"
    ") ON COMMIT PRESERVE ROWS"
)

_ATOM_STAGE_TRUNCATE = "TRUNCATE TABLE pg_temp.noesis_embedding_atom_stage"
_CONTEXT_STAGE_TRUNCATE = "TRUNCATE TABLE pg_temp.noesis_embedding_context_stage"
_SAMPLE_STAGE_TRUNCATE = "TRUNCATE TABLE pg_temp.noesis_embedding_anchor_sample_stage"
_ANCHOR_STAGE_TRUNCATE = "TRUNCATE TABLE pg_temp.noesis_embedding_anchor_stage"

_ATOM_STAGE_DROP = "DROP TABLE IF EXISTS pg_temp.noesis_embedding_atom_stage"
_CONTEXT_STAGE_DROP = "DROP TABLE IF EXISTS pg_temp.noesis_embedding_context_stage"
_SAMPLE_STAGE_DROP = "DROP TABLE IF EXISTS pg_temp.noesis_embedding_anchor_sample_stage"
_ANCHOR_STAGE_DROP = "DROP TABLE IF EXISTS pg_temp.noesis_embedding_anchor_stage"

_ATOM_STAGE_INSERT = (
    "INSERT INTO pg_temp.noesis_embedding_atom_stage (atom_id, embedding) VALUES ($1, $2::vector)"
)
_CONTEXT_STAGE_INSERT = (
    "INSERT INTO pg_temp.noesis_embedding_context_stage (event_id, predicate_pos, context_vector) "
    "VALUES ($1, $2, $3::vector)"
)
_SAMPLE_STAGE_INSERT = (
    "INSERT INTO pg_temp.noesis_embedding_anchor_sample_stage (anchor_id, event_id, predicate_pos) "
    "VALUES ($1, $2, $3) ON CONFLICT DO NOTHING"
)
_ANCHOR_STAGE_INSERT = (
    "INSERT INTO pg_temp.noesis_embedding_anchor_stage (anchor_id, centroid_vector, total_count) "
    "VALUES ($1, $2::vector, $3)"
)
_SAMPLE_REPLAY_PAGE = (
    "SELECT s.anchor_id, s.event_id, s.predicate_pos, c.context_vector::text AS context_text "
    "FROM pg_temp.noesis_embedding_anchor_sample_stage s "
    "JOIN pg_temp.noesis_embedding_context_stage c "
    "ON c.event_id = s.event_id AND c.predicate_pos = s.predicate_pos "
    "WHERE (s.anchor_id, s.event_id, s.predicate_pos) > ($1, $2, $3) "
    "ORDER BY s.anchor_id, s.event_id, s.predicate_pos LIMIT 256"
)


def _atoms_cutover_lock(schema: str) -> str:
    return f"LOCK TABLE {schema}.atoms IN SHARE ROW EXCLUSIVE MODE"


def _anchors_cutover_lock(schema: str) -> str:
    return f"LOCK TABLE {schema}.anchors IN SHARE ROW EXCLUSIVE MODE"


def _atoms_cutover_update(schema: str, null_only: bool) -> str:
    guard = "AND a.embedding IS NULL" if null_only else ""
    return (
        "UPDATE {s}.atoms a SET embedding = st.embedding "
        "FROM pg_temp.noesis_embedding_atom_stage st "
        "WHERE a.atom_id = st.atom_id AND a.status = 'active' "
        "AND a.atom_type IN ('E', 'P') {guard}"
    ).format(s=schema, guard=guard)


def _anchors_cutover_update(schema: str) -> str:
    return (
        "UPDATE {s}.anchors a SET centroid_vector = st.centroid_vector, "
        "total_count = st.total_count, updated_at = now() "
        "FROM pg_temp.noesis_embedding_anchor_stage st "
        "WHERE a.anchor_id = st.anchor_id"
    ).format(s=schema)


def _reindex_ann(schema: str) -> str:
    return f"REINDEX INDEX {schema}.{_FROZEN_ANN_INDEX}"


# ---------------------------------------------------------------------------
# ANN index classification (requirement 05A §6.12)
# ---------------------------------------------------------------------------

def _classify_ann_indexes(rows: list[Any]) -> tuple[bool, str | None]:
    """Return ``(allowed, reindex_target_name)`` for the atoms ANN indexes.

    Allowed only when there is no ANN index (target None) or exactly the frozen
    requirement-04 IVFFlat index. Any other ANN shape is a fail-closed case.
    """
    ann: list[tuple[str, str]] = []
    for row in rows:
        definition = (row["indexdef"] or "").lower()
        if "embedding" not in definition:
            continue
        if "using ivfflat" in definition or "using hnsw" in definition:
            ann.append((row["indexname"], definition))
    if not ann:
        return True, None
    if len(ann) == 1:
        name, definition = ann[0]
        if name == _FROZEN_ANN_INDEX and "using ivfflat" in definition and "vector_cosine_ops" in definition:
            return True, name
    return False, None


# ---------------------------------------------------------------------------
# State machine (requirement 05A §6.4)
# ---------------------------------------------------------------------------

async def run_embedding_rebuild(
    *,
    pool: Any,
    schema: str,
    client: Any,
    spec: BuildSpec,
    repair_null_only: bool = False,
    force_full: bool = False,
) -> int:
    """Execute the rebuild state machine; returns a process exit code (0 = ok).

    ``client`` is the injected bge seam; ``pool`` the injected asyncpg (or
    fake) pool. Exception types are never swallowed beyond converting a clear
    failure into the documented non-zero exit code.
    """
    async with pool.acquire() as conn:
        locked = await conn.fetchval("SELECT pg_try_advisory_lock($1)", _ADVISORY_LOCK_KEY)
        if not locked:
            logger.error("another noesis rebuild is already running; refusing to start")
            return _EXIT_LOCK_HELD
        try:
            return await _rebuild_locked(
                conn, schema, client, spec, repair_null_only=repair_null_only, force_full=force_full
            )
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)


async def _rebuild_locked(conn, schema, client, spec, *, repair_null_only: bool, force_full: bool) -> int:
    # ANN precheck (requirement 05A §6.12) — a full rebuild must classify the
    # atoms ANN index before touching any state. A repair never rebuilds the
    # index and therefore skips this check.
    reindex_target: str | None = None
    if not repair_null_only:
        rows = await conn.fetch(_ann_index_check(schema), schema)
        allowed, reindex_target = _classify_ann_indexes(list(rows))
        if not allowed:
            logger.error(
                "unknown/invalid ANN index shape on %s.atoms.embedding; refusing a model rebuild "
                "(expected no index or the frozen %s)",
                schema,
                _FROZEN_ANN_INDEX,
            )
            return _EXIT_ANN_UNKNOWN

    # Health-gate the client before reading profile state.
    try:
        await client.ensure_ready()
    except Exception as error:
        logger.error("embedding client not ready: %s", type(error).__name__)
        return _EXIT_CLIENT_NOT_READY

    if repair_null_only:
        return await _run_repair(conn, schema, client, spec)
    return await _run_full(conn, schema, client, spec, force_full=force_full, reindex_target=reindex_target)


async def _run_full(conn, schema, client, spec, *, force_full: bool, reindex_target: str | None) -> int:
    row = await conn.fetchrow(_profile_select(schema))
    if row is None:
        logger.error("no embedding profile row; let the online chain claim it first")
        return _EXIT_PROFILE_INVALID

    if row["status"] == "ready" and not force_full:
        if (row["model_name"], row["model_revision"], int(row["dimension"])) == (
            spec.model,
            spec.revision,
            spec.dimension,
        ):
            logger.info("profile already matches the target generation; nothing to rebuild")
            return _EXIT_OK

    # Short transaction: set rebuilding (idempotent if a prior run stopped here).
    async with conn.transaction():
        await conn.execute(_profile_set_rebuilding(schema))
    row = await conn.fetchrow(_profile_select(schema))
    if row is None or row["status"] != "rebuilding":  # pragma: no cover — just set above
        return _EXIT_PROFILE_INVALID

    return await _stage_and_cut_over(
        conn=conn,
        schema=schema,
        client=client,
        spec=spec,
        null_only=False,
        switch_profile=True,
        reindex_target=reindex_target,
    )


async def _run_repair(conn, schema, client, spec) -> int:
    row = await conn.fetchrow(_profile_select(schema))
    if row is None:
        logger.error("no embedding profile row; cannot repair without a known generation")
        return _EXIT_PROFILE_INVALID
    if row["status"] != "ready":
        logger.error("profile is %r; repair requires status=ready", row["status"])
        return _EXIT_PROFILE_INVALID
    if (row["model_name"], row["model_revision"], int(row["dimension"])) != (
        spec.model,
        spec.revision,
        spec.dimension,
    ):
        logger.error("profile generation does not match the running config; repair refused")
        return _EXIT_PROFILE_INVALID

    return await _stage_and_cut_over(
        conn=conn,
        schema=schema,
        client=client,
        spec=spec,
        null_only=True,
        switch_profile=False,
        reindex_target=None,
    )


# ---------------------------------------------------------------------------
# Staging + atomic cutover
# ---------------------------------------------------------------------------

async def _stage_and_cut_over(
    *, conn, schema, client, spec, null_only: bool, switch_profile: bool, reindex_target: str | None
) -> int:
    await conn.execute(_ATOM_STAGE_CREATE.format(dimension=int(spec.dimension)))
    await conn.execute(_ATOM_STAGE_TRUNCATE)
    drops = [_ATOM_STAGE_DROP]
    try:
        if not null_only:
            await conn.execute(_CONTEXT_STAGE_CREATE.format(dimension=int(spec.dimension)))
            await conn.execute(_SAMPLE_STAGE_CREATE.format(dimension=int(spec.dimension)))
            await conn.execute(_ANCHOR_STAGE_CREATE.format(dimension=int(spec.dimension)))
            await conn.execute(_CONTEXT_STAGE_TRUNCATE)
            await conn.execute(_SAMPLE_STAGE_TRUNCATE)
            await conn.execute(_ANCHOR_STAGE_TRUNCATE)
            drops = [_ANCHOR_STAGE_DROP, _SAMPLE_STAGE_DROP, _CONTEXT_STAGE_DROP, _ATOM_STAGE_DROP]

        code = await _stage_atoms(conn, schema, client, spec, null_only=null_only)
        if code != _EXIT_OK:
            return code

        if not null_only:
            try:
                code = await _stage_events(conn, schema, client, spec)
            except Exception as error:
                logger.error("history staging failed: %s", type(error).__name__)
                return _EXIT_HISTORY_INVALID
            if code != _EXIT_OK:
                return code
            code = await _replay_anchor_centroids(conn, spec)
            if code != _EXIT_OK:
                return code
            code = await _verify_coverage(conn, schema)
            if code != _EXIT_OK:
                return code

        return await _cut_over(
            conn,
            schema,
            spec,
            null_only=null_only,
            switch_profile=switch_profile,
            reindex_target=reindex_target,
        )
    finally:
        for statement in drops:
            await conn.execute(statement)


async def _stage_atoms(conn, schema, client, spec, *, null_only: bool) -> int:
    """Encode every active E/P atom to the target generation into the atom stage."""
    last_atom_id = 0
    while True:
        page = await conn.fetch(_atoms_page(schema, null_only=null_only), last_atom_id)
        if not page:
            return _EXIT_OK
        for atom in page:
            atom_id = int(atom["atom_id"])
            text = atom["text"]
            atom_type = atom["atom_type"]
            try:
                vector = await client.embed_identity(text, atom_type)
            except Exception as error:
                logger.error("atom encoding failed for atom %s (%s): %s", atom_id, atom_type, type(error).__name__)
                return _EXIT_ENCODE_FAILED
            if len(vector) != spec.dimension or not all(_is_finite(component) for component in vector):
                logger.error(
                    "atom %s encoded to %d dims (expected %d) or produced a non-finite component",
                    atom_id,
                    len(vector),
                    spec.dimension,
                )
                return _EXIT_ENCODE_FAILED
            await conn.execute(_ATOM_STAGE_INSERT, atom_id, vector_literal(vector))
            last_atom_id = atom_id


async def _stage_events(conn, schema, client, spec) -> int:
    """Rebuild context/sample stages from the authoritative events.data component."""
    last_event_id = 0
    while True:
        page = await conn.fetch(_events_page(schema), last_event_id)
        if not page:
            return _EXIT_OK
        for event in page:
            event_id = int(event["event_id"])
            data = event["data"]
            code = await _stage_one_event(conn, schema, client, event_id, data)
            if code != _EXIT_OK:
                return code
            last_event_id = event_id


async def _stage_one_event(conn, schema, client, event_id: int, data: Any) -> int:
    component_atoms = _component_atoms(event_id, data)
    if component_atoms is None:
        return _EXIT_HISTORY_INVALID

    try:
        plan = plan_anchor_routing(component_atoms)
    except AnchorFrameError as error:
        logger.error("event %s component closure is not rebuildable: %s", event_id, error)
        return _EXIT_HISTORY_INVALID

    db_rows = await conn.fetch(_event_atoms_for_event(schema), event_id)
    if not _event_atoms_match_component(event_id, list(db_rows), plan):
        return _EXIT_HISTORY_INVALID

    atom_ids = sorted({int(row["atom_id"]) for row in db_rows})
    atom_rows = await conn.fetch(_atoms_by_ids(schema), atom_ids)
    by_id = {int(row["atom_id"]): row for row in atom_rows}
    if len(by_id) != len(atom_ids):
        logger.error("event %s references atoms that do not exist", event_id)
        return _EXIT_HISTORY_INVALID

    occurrence_frame = {occurrence.pos: occurrence.frame_pos for occurrence in plan.occurrences}

    # Cross-validate literal -> atom_id and anchor -> atom_id (§4.4 #5/#7/#8).
    anchor_ids = sorted({int(row["anchor_id"]) for row in db_rows})
    anchor_rows = await conn.fetch(_anchors_by_ids(schema), anchor_ids)
    anchor_atom = {int(row["anchor_id"]): int(row["atom_id"]) for row in anchor_rows}
    if len(anchor_atom) != len(anchor_ids):
        logger.error("event %s references anchors that do not exist", event_id)
        return _EXIT_HISTORY_INVALID

    frame_anchor: dict[tuple[int, int], int] = {}
    for row in db_rows:
        atom_id = int(row["atom_id"])
        occurrence_id = int(row["occurrence_id"])
        anchor_id = int(row["anchor_id"])
        atom_row = by_id[atom_id]
        text = atom_row["text"]
        atom_type = atom_row["atom_type"]
        if atom_type not in ("E", "P"):
            logger.error("event %s atom %s has non-routable type %r", event_id, atom_id, atom_type)
            return _EXIT_HISTORY_INVALID
        if anchor_atom.get(anchor_id) != atom_id:
            logger.error("event %s anchor %s does not belong to atom %s", event_id, anchor_id, atom_id)
            return _EXIT_HISTORY_INVALID
        component_atom = next((a for a in component_atoms if a.pos == occurrence_id), None)
        if component_atom is None:
            logger.error("event %s db occurrence %s is absent from the component", event_id, occurrence_id)
            return _EXIT_HISTORY_INVALID
        if (component_atom.text, component_atom.type) != (text, atom_type):
            logger.error("event %s occurrence %s literal does not match its atom row", event_id, occurrence_id)
            return _EXIT_HISTORY_INVALID
        frame_pos = occurrence_frame.get(occurrence_id)
        if frame_pos is None:
            logger.error("event %s occurrence %s has no predicate frame", event_id, occurrence_id)
            return _EXIT_HISTORY_INVALID
        key = (atom_id, frame_pos)
        if key in frame_anchor and frame_anchor[key] != anchor_id:
            logger.error("event %s atom %s frame %s maps to two anchors", event_id, atom_id, frame_pos)
            return _EXIT_HISTORY_INVALID
        frame_anchor[key] = anchor_id

    # One Context Vector per (event, predicate frame), embedded once (§6.9.3).
    if not await _embed_contexts(conn, client, event_id, plan):
        return _EXIT_ENCODE_FAILED

    for (atom_id, frame_pos), anchor_id in sorted(frame_anchor.items()):
        await conn.execute(_SAMPLE_STAGE_INSERT, anchor_id, event_id, frame_pos)
    return _EXIT_OK


async def _embed_contexts(conn, client, event_id: int, plan) -> bool:
    for predicate_pos in sorted(plan.frames):
        context_text = plan.frames[predicate_pos].context_text
        try:
            vector = await client.embed_context(context_text)
        except Exception as error:
            logger.error(
                "context encoding failed for event %s predicate %s: %s", event_id, predicate_pos, type(error).__name__
            )
            return False
        if len(vector) != 1024 or not all(_is_finite(component) for component in vector):
            logger.error("event %s predicate %s produced a bad context vector", event_id, predicate_pos)
            return False
        await conn.execute(_CONTEXT_STAGE_INSERT, event_id, predicate_pos, vector_literal(vector))
    return True


async def _replay_anchor_centroids(conn, spec) -> int:
    """Replay the frozen EMA in bounded keyset pages.

    Only the current Anchor centroid/count and one 256-row database page are
    retained in memory.  The cursor key and accumulator deliberately survive
    page boundaries so an Anchor with more than one page of samples gets the
    exact same deterministic EMA as an unpaged ordered stream.
    """
    current_anchor: int | None = None
    centroid: list[float] | None = None
    count = 0
    last_key = (-1, -1, -1)
    while True:
        rows = await conn.fetch(_SAMPLE_REPLAY_PAGE, *last_key)
        if not rows:
            break
        for row in rows:
            anchor_id = int(row["anchor_id"])
            event_id = int(row["event_id"])
            predicate_pos = int(row["predicate_pos"])
            context = parse_centroid_text(row["context_text"], dimension=spec.dimension)
            if anchor_id != current_anchor:
                if current_anchor is not None and centroid is not None:
                    await conn.execute(_ANCHOR_STAGE_INSERT, current_anchor, vector_literal(centroid), count)
                current_anchor = anchor_id
                centroid = list(context)
                count = 1
            else:
                centroid = ema_centroid(centroid, context)
                count += 1
            last_key = (anchor_id, event_id, predicate_pos)
    if current_anchor is not None and centroid is not None:
        await conn.execute(_ANCHOR_STAGE_INSERT, current_anchor, vector_literal(centroid), count)
    return _EXIT_OK


async def _verify_coverage(conn, schema) -> int:
    active_atoms = int(await conn.fetchval(_active_ep_count(schema)))
    staged_atoms = int(await conn.fetchval(_atom_stage_count(schema)))
    if active_atoms != staged_atoms:
        logger.error("atom stage coverage incomplete: %d staged vs %d active", staged_atoms, active_atoms)
        return _EXIT_COVERAGE_INCOMPLETE
    all_anchors = int(await conn.fetchval(_all_anchor_count(schema)))
    staged_anchors = int(await conn.fetchval(_anchor_stage_count(schema)))
    if all_anchors != staged_anchors:
        orphans = await conn.fetch(_orphan_anchors(schema))
        logger.error(
            "anchor stage coverage incomplete: %d staged vs %d anchors (%d orphan(s) shown)",
            staged_anchors,
            all_anchors,
            len(orphans),
        )
        for orphan in orphans:
            logger.error("orphan anchor %s has no rebuildable sample (atom %s)", orphan["anchor_id"], orphan["atom_id"])
        return _EXIT_COVERAGE_INCOMPLETE
    return _EXIT_OK


async def _cut_over(
    conn, schema, spec, *, null_only: bool, switch_profile: bool, reindex_target: str | None
) -> int:
    try:
        async with conn.transaction():
            if switch_profile:
                row = await conn.fetchrow(_profile_select(schema))
                if row is None or row["status"] != "rebuilding":
                    logger.error("profile changed mid-rebuild; aborting final commit")
                    return _EXIT_CUTOVER_FAILED
            await conn.execute(_atoms_cutover_lock(schema))
            code = await _verify_coverage(conn, schema) if not null_only else _EXIT_OK
            if code != _EXIT_OK:
                return code
            if not null_only:
                await conn.execute(_anchors_cutover_lock(schema))
            await conn.execute(_atoms_cutover_update(schema, null_only=null_only))
            if not null_only:
                await conn.execute(_anchors_cutover_update(schema))
                if reindex_target is not None:
                    await conn.execute(_reindex_ann(schema))
            if switch_profile:
                await conn.execute(
                    _profile_set_ready(schema), spec.model, spec.revision, spec.dimension
                )
    except Exception as error:
        logger.error("final cutover failed and rolled back: %s", type(error).__name__)
        return _EXIT_CUTOVER_FAILED
    action = "repaired" if null_only else "rebuilt"
    logger.info("%s embedding space toward generation %s/%s/%d", action, spec.model, spec.revision, spec.dimension)
    return _EXIT_OK


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _is_finite(value: Any) -> bool:
    import math

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _component_atoms(event_id: int, data: Any) -> list[Any] | None:
    """Parse the authoritative ``events.data.component`` atoms (§4.3/§4.4 #1).

    asyncpg hands back ``JSONB`` as a string unless a codec is registered, so a
    string payload is decoded here before the object check.
    """
    from hyperextract.noesis import FactComponent

    if isinstance(data, (str, bytes, bytearray)):
        try:
            data = json.loads(data)
        except Exception:
            logger.error("event %s data is not valid JSON", event_id)
            return None
    if not isinstance(data, dict):
        logger.error("event %s data is not an object", event_id)
        return None
    component_json = data.get("component")
    if not isinstance(component_json, dict):
        logger.error("event %s data has no component object", event_id)
        return None
    try:
        component = FactComponent.model_validate(component_json)
    except Exception as error:
        logger.error("event %s component failed the closure model: %s", event_id, type(error).__name__)
        return None
    return list(component.atoms)


def _event_atoms_match_component(event_id: int, db_rows: list[Any], plan) -> bool:
    """Occurrence ids must match the component pos set exactly (§4.4 #4/#9)."""
    db_occurrences = {int(row["occurrence_id"]) for row in db_rows}
    component_positions = {occurrence.pos for occurrence in plan.occurrences}
    if db_occurrences != component_positions:
        logger.error(
            "event %s occurrence set mismatch: db=%d component=%d",
            event_id,
            len(db_occurrences),
            len(component_positions),
        )
        return False
    for row in db_rows:
        if row["anchor_id"] is None:
            logger.error("event %s occurrence %s has a NULL anchor", event_id, row["occurrence_id"])
            return False
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_client(config: Any) -> Any:
    from .noesis_embedding import NoesisEmbeddingClient

    return NoesisEmbeddingClient(
        base_url=config.noesis_embedding_base_url,
        model=config.noesis_embedding_model,
        revision=config.noesis_embedding_revision,
        dimension=int(config.noesis_embedding_dimension),
        timeout_seconds=float(config.noesis_embedding_timeout_seconds),
        max_retries=int(config.noesis_embedding_max_retries),
        api_key=getattr(config, "noesis_embedding_api_key", "") or "",
    )


def _load_config() -> Any:
    from ...config import HindsightConfig, load_dotenv_for_entrypoint

    load_dotenv_for_entrypoint()
    return HindsightConfig.from_env()


async def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Noesis shared embedding-space rebuild")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--repair-null-only",
        action="store_true",
        help="only fill NULL E/P embeddings under the current ready generation; "
        "never touch anchors, the ANN index, or the profile, and never scan events",
    )
    mode.add_argument(
        "--force-full",
        action="store_true",
        help="run a full rebuild even when the profile already equals the target "
        "generation; never bypasses health, lock, coverage, or rollback checks",
    )
    args = parser.parse_args(argv)

    config = _load_config()
    schema = getattr(config, "noesis_schema", "noesis_core") or "noesis_core"
    spec = BuildSpec(
        model=config.noesis_embedding_model,
        revision=config.noesis_embedding_revision,
        dimension=int(config.noesis_embedding_dimension),
    )
    client = _build_client(config)
    pool = await asyncpg.create_pool(dsn=config.noesis_database_url)
    try:
        return await run_embedding_rebuild(
            pool=pool,
            schema=schema,
            client=client,
            spec=spec,
            repair_null_only=args.repair_null_only,
            force_full=args.force_full,
        )
    finally:
        await pool.close()
        await client.aclose()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
