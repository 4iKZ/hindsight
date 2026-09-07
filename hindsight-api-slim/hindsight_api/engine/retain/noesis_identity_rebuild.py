"""Noesis identity rebuild command (requirement 03 §10.4).

A controlled, profile-aware full recompute of the identity vectors for
``noesis_core.atoms``, so switching to a new (model, revision, dimension)
generation cannot silently mix two model spaces in ``atoms.embedding``.

Run modes:

* default  — full recompute: flip ``embedding_profiles.identity`` to
  ``rebuilding``, encode every active E/P atom **outside** the transaction
  into an in-process staging dict (never touching ``atoms.embedding``), and
  only after every atom succeeds does one final short transaction atomically
  apply all vectors and flip the profile to ``ready`` (new generation).
* ``--repair-null-only`` — profile must already match the running config and
  be ``ready``; only NULL E/P embeddings are filled in, and the profile is
  left untouched.

Invariants (§10.4 / §15.5):

* a PostgreSQL session advisory lock guarantees one rebuild at a time; a
  second instance fails clearly instead of running in parallel;
* encoding never mutates ``atoms.embedding`` mid-run: any failure exits
  non-zero leaving all live vectors untouched (profile stays ``rebuilding``,
  retryable and idempotent);
* the final switch is a single short transaction re-locking the profile,
  verifying it is still ``rebuilding``, then updating all E/P and setting the
  new generation + ``ready`` atomically;
* an existing ANN (ivfflat/hnsw) index on ``atoms`` refuses a full rebuild —
  the caller must run the requirement-04-aware index rebuild instead.

The CLI entry point builds a real client (``NoesisIdentityClient``) and a real
asyncpg pool from the Hindsight config; the async core is injected with a
client and a pool so tests can run fully offline against fake seams.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

# Session advisory lock key (arbitrary but stable) used to serialize rebuilds.
_ADVISORY_LOCK_KEY = 0x4E4F4553_495352 # "NOESISR"

_SPEC = "identity"


@dataclass(frozen=True)
class BuildSpec:
    """The target (model, revision, dimension) generation to rebuild toward."""

    model: str
    revision: str
    dimension: int


def _profile_select(schema: str) -> str:
    return (
        "SELECT model_name, model_revision, dimension, status FROM {s}.embedding_profiles "
        "WHERE embedding_kind = 'identity' FOR UPDATE"
    ).format(s=schema)


def _profile_set_rebuilding(schema: str) -> str:
    return (
        "UPDATE {s}.embedding_profiles SET status = 'rebuilding', updated_at = now() "
        "WHERE embedding_kind = 'identity'"
    ).format(s=schema)


def _profile_set_ready(schema: str) -> str:
    return (
        "UPDATE {s}.embedding_profiles SET model_name = $1, model_revision = $2, "
        "dimension = $3, status = 'ready', updated_at = now() WHERE embedding_kind = 'identity'"
    ).format(s=schema)


def _atoms_missing_stage(schema: str, null_only: bool) -> str:
    guard = "AND a.embedding IS NULL " if null_only else ""
    return (
        "SELECT a.atom_id, a.text, a.atom_type FROM {s}.atoms a "
        "LEFT JOIN pg_temp.noesis_identity_rebuild_stage st ON st.atom_id = a.atom_id "
        "WHERE a.status = 'active' AND a.atom_type IN ('E', 'P') {guard}"
        "AND st.atom_id IS NULL ORDER BY a.atom_id"
    ).format(s=schema, guard=guard)


def _stage_create(dimension: int) -> str:
    return (
        "CREATE TEMP TABLE IF NOT EXISTS noesis_identity_rebuild_stage ("
        "atom_id BIGINT PRIMARY KEY, embedding VECTOR({dimension}) NOT NULL"
        ") ON COMMIT PRESERVE ROWS"
    ).format(dimension=int(dimension))


_STAGE_TRUNCATE = "TRUNCATE TABLE pg_temp.noesis_identity_rebuild_stage"
_STAGE_DROP = "DROP TABLE IF EXISTS pg_temp.noesis_identity_rebuild_stage"
_STAGE_INSERT = (
    "INSERT INTO pg_temp.noesis_identity_rebuild_stage (atom_id, embedding) "
    "VALUES ($1, $2::vector)"
)


def _atoms_cutover_update(schema: str, null_only: bool) -> str:
    guard = "AND a.embedding IS NULL" if null_only else ""
    return (
        "UPDATE {s}.atoms a SET embedding = st.embedding "
        "FROM pg_temp.noesis_identity_rebuild_stage st "
        "WHERE a.atom_id = st.atom_id AND a.status = 'active' "
        "AND a.atom_type IN ('E', 'P') {guard}"
    ).format(
        s=schema, guard=guard
    )


def _atoms_cutover_lock(schema: str) -> str:
    return f"LOCK TABLE {schema}.atoms IN SHARE ROW EXCLUSIVE MODE"


def _ann_index_check(schema: str) -> str:
    return (
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = $1 AND tablename = 'atoms'"
    )


def _vector_literal(vector: Any) -> str:
    return "[" + ",".join(repr(float(component)) for component in vector) + "]"


async def run_identity_rebuild(
    *,
    pool: Any,
    schema: str,
    client: Any,
    spec: BuildSpec,
    repair_null_only: bool = False,
) -> int:
    """Execute the rebuild state machine; returns a process exit code (0 = ok).

    ``client`` is the injected bge seam; ``pool`` the injected asyncpg (or
    fake) pool. Exception types are never swallowed here beyond converting a
    clear failure into a non-zero exit code — the CLI turns them into a
    message.
    """
    # 1. Serialize: only one rebuild at a time (session advisory lock).
    async with pool.acquire() as conn:
        locked = await conn.fetchval("SELECT pg_try_advisory_lock($1)", _ADVISORY_LOCK_KEY)
        if not locked:
            logger.error("another identity rebuild is already running; refusing to start")
            return 2
        try:
            return await _rebuild_locked(conn, schema, client, spec, repair_null_only)
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)


async def _rebuild_locked(conn, schema, client, spec, repair_null_only) -> int:
    # 2. Refuse a full rebuild if an ANN index already exists on atoms.
    if not repair_null_only:
        indexes = await conn.fetch(_ann_index_check(schema), schema)
        ann = [row for row in indexes if "embedding" in row["indexdef"]]
        if ann:
            logger.error(
                "an ANN index on atoms.embedding exists (%s); refusing a full rebuild — "
                "use requirement-04's index-aware rebuild instead",
                ann[0]["indexname"],
            )
            return 3

    # 3. Health-gate the client before reading profile state.
    try:
        await client.ensure_ready()
    except Exception as error:  # config invalid / service unavailable
        logger.error("identity client not ready: %s", type(error).__name__)
        return 4

    if repair_null_only:
        return await _run_repair(conn, schema, client, spec)
    return await _run_full(conn, schema, client, spec)


async def _run_full(conn, schema, client, spec) -> int:
    # Verify a profile row exists: the online claim path must have created it.
    row = await conn.fetchrow(_profile_select(schema))
    if row is None:
        logger.error("no identity profile row; let the online chain claim it first")
        return 5
    current_status = row["status"]

    # If it is already rebuilding, we are resuming an interrupted run; if it is
    # ready and matches the target, there is nothing to do.
    if current_status == "ready":
        if (row["model_name"], row["model_revision"], int(row["dimension"])) == (
            spec.model,
            spec.revision,
            spec.dimension,
        ):
            logger.info("profile already matches the target generation; nothing to rebuild")
            return 0

    # Short transaction: set rebuilding (idempotent if a prior run stopped here).
    async with conn.transaction():
        await conn.execute(_profile_set_rebuilding(schema))
    # Re-read to confirm the state we expect before running the long encode.
    row = await conn.fetchrow(_profile_select(schema))
    if row is None:  # pragma: no cover — defended above
        return 5

    return await _stage_and_cut_over(
        conn=conn,
        schema=schema,
        client=client,
        spec=spec,
        null_only=False,
        switch_profile=True,
    )


async def _run_repair(conn, schema, client, spec) -> int:
    # Profile must already match the running config AND be ready.
    row = await conn.fetchrow(_profile_select(schema))
    if row is None:
        logger.error("no identity profile row; cannot repair without a known generation")
        return 5
    if row["status"] != "ready":
        logger.error("profile is %r; repair requires status=ready", row["status"])
        return 5
    if (row["model_name"], row["model_revision"], int(row["dimension"])) != (
        spec.model,
        spec.revision,
        spec.dimension,
    ):
        logger.error("profile generation does not match the running config; repair refused")
        return 5

    return await _stage_and_cut_over(
        conn=conn,
        schema=schema,
        client=client,
        spec=spec,
        null_only=True,
        switch_profile=False,
    )


async def _stage_and_cut_over(
    *, conn: Any, schema: str, client: Any, spec: BuildSpec, null_only: bool, switch_profile: bool
) -> int:
    """Encode into a bounded DB TEMP stage, then perform one atomic cutover.

    Online facts continue while encoding. Before cutover we briefly lock the
    live atoms table; if an E/P atom arrived since the last scan, the lock is
    released, that delta is encoded outside a transaction, and cutover retries.
    """
    encoded = 0
    await conn.execute(_stage_create(spec.dimension))
    await conn.execute(_STAGE_TRUNCATE)
    try:
        while True:
            targets = await conn.fetch(_atoms_missing_stage(schema, null_only=null_only))
            if targets:
                for target in targets:
                    try:
                        vector = await client.embed(target["text"], target["atom_type"])
                    except Exception as error:
                        logger.error(
                            "encoding failed for atom %s (%s/%s): %s",
                            target["atom_id"],
                            target["text"],
                            target["atom_type"],
                            type(error).__name__,
                        )
                        return 6
                    if len(vector) != spec.dimension:  # pragma: no cover — client enforces
                        logger.error(
                            "atom %s encoded to %d dims, expected %d",
                            target["atom_id"],
                            len(vector),
                            spec.dimension,
                        )
                        return 6
                    await conn.execute(_STAGE_INSERT, target["atom_id"], _vector_literal(vector))
                    encoded += 1
                continue

            late_targets: list[Any] = []
            async with conn.transaction():
                await conn.execute(_atoms_cutover_lock(schema))
                late_targets = await conn.fetch(_atoms_missing_stage(schema, null_only=null_only))
                if not late_targets:
                    if switch_profile:
                        row = await conn.fetchrow(_profile_select(schema))
                        if row is None or row["status"] != "rebuilding":
                            logger.error("profile changed mid-rebuild; aborting final commit")
                            return 7
                    await conn.execute(_atoms_cutover_update(schema, null_only=null_only))
                    if switch_profile:
                        await conn.execute(
                            _profile_set_ready(schema),
                            spec.model,
                            spec.revision,
                            spec.dimension,
                        )
            if late_targets:
                continue
            action = "rebuilt" if switch_profile else "repaired"
            logger.info("%s %d identity vectors", action, encoded)
            return 0
    finally:
        await conn.execute(_STAGE_DROP)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_client(config: Any) -> Any:
    from .noesis_identity_vector import NoesisIdentityClient

    return NoesisIdentityClient(
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

    parser = argparse.ArgumentParser(description="Noesis identity vector rebuild")
    parser.add_argument(
        "--repair-null-only",
        action="store_true",
        help="only fill NULL E/P embeddings using the current profile generation; "
        "never overwrite existing vectors, never change the profile",
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
        return await run_identity_rebuild(
            pool=pool, schema=schema, client=client, spec=spec, repair_null_only=args.repair_null_only
        )
    finally:
        await pool.close()
        await client.aclose()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
