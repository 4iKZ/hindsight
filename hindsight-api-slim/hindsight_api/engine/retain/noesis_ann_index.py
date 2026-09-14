"""Noesis ANN IVFFlat index lifecycle (requirement 04 Tasks 6–7).

status / build / reindex / drop for the single frozen index
``idx_atoms_embedding_ivfflat``. Concurrent DDL is never run inside a
transaction block. The CLI reuses HindsightConfig and must not invent a
second .env lookup.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from .noesis_ingest import validate_schema_identifier

logger = logging.getLogger(__name__)

INDEX_NAME = "idx_atoms_embedding_ivfflat"
_ELIGIBLE_THRESHOLD = 10_000
# Dedicated session key; must not equal the identity-rebuild lock.
_ADVISORY_LOCK_KEY = 0x4E4F4553_041446  # "NOESIS" + 04 IVF

_CATALOG_SELECT = """
SELECT
    n.nspname AS schemaname,
    t.relname AS tablename,
    i.relname AS indexname,
    am.amname,
    x.indisvalid,
    x.indisready,
    pg_get_indexdef(i.oid) AS indexdef
FROM pg_class AS i
JOIN pg_namespace AS n ON n.oid = i.relnamespace
JOIN pg_index AS x ON x.indexrelid = i.oid
JOIN pg_class AS t ON t.oid = x.indrelid
JOIN pg_am AS am ON am.oid = i.relam
WHERE n.nspname = $1
  AND t.relname = 'atoms'
  AND i.relname = '{index}'
""".replace("{index}", INDEX_NAME)

_ELIGIBLE_COUNT = (
    "SELECT count(*) FROM {s}.atoms "
    "WHERE status = 'active' AND atom_type IN ('E', 'P') AND embedding IS NOT NULL"
)

_CREATE_INDEX = (
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS {index} "
    "ON {s}.atoms USING ivfflat (embedding vector_cosine_ops) "
    "WITH (lists = 100) "
    "WHERE status = 'active' AND atom_type IN ('E', 'P') AND embedding IS NOT NULL"
).replace("{index}", INDEX_NAME)

_REINDEX = "REINDEX INDEX CONCURRENTLY {s}.{index}".replace("{index}", INDEX_NAME)
_DROP_INDEX = "DROP INDEX CONCURRENTLY IF EXISTS {s}.{index}".replace("{index}", INDEX_NAME)
_ANALYZE = "ANALYZE {s}.atoms"


@dataclass(frozen=True)
class AnnIndexStatus:
    exists: bool
    valid: bool
    ready: bool
    definition: str | None
    eligible_rows: int


class AnnIndexError(Exception):
    """Base class for ANN index lifecycle failures."""


class AnnIndexConflict(AnnIndexError):
    """Same name exists but the definition is not the frozen IVFFlat."""


class AnnIndexBusy(AnnIndexError):
    """Another index manager holds the advisory lock."""


class AnnIndexMissing(AnnIndexError):
    """reindex requires the frozen index to exist."""


def _sql(schema: str, statement: str) -> str:
    return statement.format(s=schema)


def _require_schema(schema: str) -> str:
    if not validate_schema_identifier(schema):
        raise ValueError(f"invalid noesis schema identifier: {schema!r}")
    return schema


def _row_value(row: Any, key: str) -> Any:
    return row[key]


def definition_is_frozen(definition: str | None, *, schema: str) -> bool:
    if not definition:
        return False
    return (
        INDEX_NAME in definition
        and schema in definition
        and "ivfflat" in definition.lower()
        and "vector_cosine_ops" in definition
        and ("lists='100'" in definition or "lists = 100" in definition)
        and "active" in definition
        and "atom_type" in definition
        and "embedding IS NOT NULL" in definition
    )


def status_is_healthy(status: AnnIndexStatus, *, schema: str) -> bool:
    return (
        status.exists
        and status.valid
        and status.ready
        and definition_is_frozen(status.definition, schema=schema)
    )


async def get_ann_index_status(conn, *, schema: str) -> AnnIndexStatus:
    schema = _require_schema(schema)
    eligible = await conn.fetchval(_sql(schema, _ELIGIBLE_COUNT))
    row = await conn.fetchrow(_CATALOG_SELECT, schema)
    if row is None:
        return AnnIndexStatus(
            exists=False,
            valid=False,
            ready=False,
            definition=None,
            eligible_rows=int(eligible or 0),
        )
    definition = _row_value(row, "indexdef")
    exists = str(_row_value(row, "indexname") or "") == INDEX_NAME
    return AnnIndexStatus(
        exists=exists,
        valid=bool(_row_value(row, "indisvalid")),
        ready=bool(_row_value(row, "indisready")),
        definition=None if definition is None else str(definition),
        eligible_rows=int(eligible or 0),
    )


async def _hold_lock(conn) -> None:
    locked = await conn.fetchval("SELECT pg_try_advisory_lock($1)", _ADVISORY_LOCK_KEY)
    if not locked:
        raise AnnIndexBusy("another ANN index manager is already running")


async def _release_lock(conn) -> None:
    await conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)


async def build_ann_index(conn, *, schema: str, force: bool = False) -> bool:
    schema = _require_schema(schema)
    await _hold_lock(conn)
    try:
        status = await get_ann_index_status(conn, schema=schema)
        if status_is_healthy(status, schema=schema):
            return False
        if status.exists and not definition_is_frozen(status.definition, schema=schema):
            raise AnnIndexConflict(
                f"{INDEX_NAME} exists with a non-frozen definition; report to Leader"
            )
        if status.eligible_rows < _ELIGIBLE_THRESHOLD and not force:
            logger.info("ann index build cold_start eligible_rows=%s forced=%s", status.eligible_rows, False)
            return False
        if force:
            logger.info("ann index build forced=true eligible_rows=%s", status.eligible_rows)
        await conn.execute(_sql(schema, _CREATE_INDEX))
        await conn.execute(_sql(schema, _ANALYZE))
        rebuilt = await get_ann_index_status(conn, schema=schema)
        return status_is_healthy(rebuilt, schema=schema)
    finally:
        await _release_lock(conn)


async def reindex_ann_index(conn, *, schema: str) -> None:
    schema = _require_schema(schema)
    await _hold_lock(conn)
    try:
        status = await get_ann_index_status(conn, schema=schema)
        if not status.exists:
            raise AnnIndexMissing(f"{INDEX_NAME} does not exist; refuse silent build")
        await conn.execute(_sql(schema, _REINDEX))
        await conn.execute(_sql(schema, _ANALYZE))
    finally:
        await _release_lock(conn)


async def drop_ann_index(conn, *, schema: str) -> None:
    schema = _require_schema(schema)
    await _hold_lock(conn)
    try:
        await conn.execute(_sql(schema, _DROP_INDEX))
    finally:
        await _release_lock(conn)


def _load_config() -> Any:
    from ...config import HindsightConfig, load_dotenv_for_entrypoint

    load_dotenv_for_entrypoint()
    return HindsightConfig.from_env()


async def _connect(config: Any):
    import asyncpg

    return await asyncpg.connect(dsn=config.noesis_database_url)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Noesis ANN IVFFlat index manager")
    parser.add_argument("action", choices=("status", "build", "reindex", "drop"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="build only: bypass the 10,000-row cold-start threshold",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        raise ValueError("illegal CLI arguments") from error
    if args.force and args.action != "build":
        raise ValueError("--force is accepted only by build")
    return args


async def _run_action(conn, *, schema: str, action: str, force: bool) -> int:
    if action == "status":
        status = await get_ann_index_status(conn, schema=schema)
        logger.info(
            "ann index status exists=%s valid=%s ready=%s eligible_rows=%s",
            status.exists,
            status.valid,
            status.ready,
            status.eligible_rows,
        )
        if status.exists and not status_is_healthy(status, schema=schema):
            return 3
        return 0
    if action == "build":
        await build_ann_index(conn, schema=schema, force=force)
        return 0
    if action == "reindex":
        await reindex_ann_index(conn, schema=schema)
        return 0
    await drop_ann_index(conn, schema=schema)
    return 0


async def _main(
    argv: list[str] | None = None,
    *,
    connect: Callable[[Any], Any] | None = None,
    config: Any | None = None,
) -> int:
    try:
        args = _parse_args(argv)
    except ValueError:
        return 2
    try:
        loaded = config if config is not None else _load_config()
        schema = getattr(loaded, "noesis_schema", None) or "noesis_core"
        if not validate_schema_identifier(schema):
            return 2
        if not getattr(loaded, "noesis_database_url", None):
            return 2
    except Exception:
        return 2

    connector = connect or _connect
    conn = None
    try:
        conn = await connector(loaded)
        return await _run_action(conn, schema=schema, action=args.action, force=args.force)
    except AnnIndexConflict:
        return 3
    except AnnIndexError:
        return 4
    except Exception:
        logger.exception("ann index CLI failed")
        return 4
    finally:
        if conn is not None:
            closer = getattr(conn, "close", None)
            if closer is not None:
                result = closer()
                if hasattr(result, "__await__"):
                    await result


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
