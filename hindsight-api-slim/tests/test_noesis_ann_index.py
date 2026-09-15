"""Noesis ANN IVFFlat index lifecycle contract tests (requirement 04 Task 5).

Offline only: a scriptable catalog fake. No real PostgreSQL. Task 5 writes
failing tests for status/build/reindex/drop. The implementation module is
Task 6.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hindsight_api.engine.retain.noesis_ann_index import (
    _MODEL_SWITCH_EPILOG,
    AnnIndexBusy,
    AnnIndexConflict,
    AnnIndexError,
    AnnIndexMissing,
    AnnIndexStatus,
    build_ann_index,
    drop_ann_index,
    get_ann_index_status,
    reindex_ann_index,
)

INDEX_NAME = "idx_atoms_embedding_ivfflat"
THRESHOLD = 10_000


def _healthy_definition(schema: str = "noesis_core") -> str:
    return (
        f"CREATE INDEX {INDEX_NAME} ON {schema}.atoms USING ivfflat "
        "(embedding vector_cosine_ops) WITH (lists='100') "
        "WHERE ((status = 'A') AND (atom_type IN ('E', 'P')) "
        "AND (embedding IS NOT NULL))"
    )


def _catalog(*, definition=None, amname="ivfflat", valid=True, ready=True, schema="noesis_core"):
    return {
        "schemaname": schema,
        "tablename": "atoms",
        "indexname": INDEX_NAME,
        "amname": amname,
        "indisvalid": valid,
        "indisready": ready,
        "indexdef": definition if definition is not None else _healthy_definition(schema),
    }


def _status_is_healthy(status: AnnIndexStatus, schema: str = "noesis_core") -> bool:
    definition = status.definition or ""
    return (
        status.exists
        and status.valid
        and status.ready
        and "vector_cosine_ops" in definition
        and ("lists='100'" in definition or "lists = 100" in definition)
        and "status" in definition
        and "status = 'A'" in definition
        and "atom_type" in definition
        and "embedding IS NOT NULL" in definition
        and INDEX_NAME in definition
        and schema in definition
    )


class _IndexConn:
    """Catalog + advisory-lock fake for the index manager."""

    def __init__(
        self,
        *,
        catalog=None,
        eligible_rows: int = 0,
        lock_available: bool = True,
        create_valid: bool = True,
        create_ready: bool = True,
    ) -> None:
        self.catalog = None if catalog is None else dict(catalog)
        self.eligible_rows = eligible_rows
        self.lock_available = lock_available
        self.create_valid = create_valid
        self.create_ready = create_ready
        self.lock_held = False
        self.calls: list[tuple[str, str, tuple]] = []
        self.analyzed = False
        self.transaction_used = False
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    def transaction(self, **kwargs):
        self.transaction_used = True
        raise AssertionError("CONCURRENTLY DDL must not run in a transaction block")

    async def fetchval(self, sql: str, *args):
        self.calls.append(("fetchval", sql, args))
        if "pg_try_advisory_lock" in sql:
            if not self.lock_available or self.lock_held:
                return False
            self.lock_held = True
            return True
        if "count(*)" in sql.lower():
            return self.eligible_rows
        raise AssertionError(f"unexpected fetchval SQL: {sql}")

    async def fetchrow(self, sql: str, *args):
        self.calls.append(("fetchrow", sql, args))
        if self.catalog is None:
            return None
        return dict(self.catalog)

    async def fetch(self, sql: str, *args):
        self.calls.append(("fetch", sql, args))
        if self.catalog is None:
            return []
        return [dict(self.catalog)]

    async def execute(self, sql: str, *args):
        self.calls.append(("execute", sql, args))
        upper = sql.lstrip().upper()
        if "PG_ADVISORY_UNLOCK" in upper:
            self.lock_held = False
            return "SELECT 1"
        if "CREATE INDEX" in upper:
            if "CONCURRENTLY" not in upper:
                raise AssertionError("build must use CREATE INDEX CONCURRENTLY")
            if INDEX_NAME not in sql:
                raise AssertionError("build must use the frozen index name")
            self.catalog = _catalog(valid=self.create_valid, ready=self.create_ready)
            return "CREATE INDEX"
        if upper.startswith("REINDEX"):
            if "CONCURRENTLY" not in upper:
                raise AssertionError("reindex must use REINDEX INDEX CONCURRENTLY")
            if self.catalog is None:
                raise RuntimeError("index does not exist")
            return "REINDEX"
        if "DROP INDEX" in upper:
            if "CONCURRENTLY" not in upper:
                raise AssertionError("drop must use DROP INDEX CONCURRENTLY")
            self.catalog = None
            return "DROP INDEX"
        if upper.startswith("ANALYZE"):
            self.analyzed = True
            return "ANALYZE"
        raise AssertionError(f"unexpected execute SQL: {sql}")


def _sqls(conn: _IndexConn) -> list[str]:
    return [sql for _method, sql, _args in conn.calls]


# ---------------------------------------------------------------------------
# AnnIndexStatus
# ---------------------------------------------------------------------------


def test_ann_index_status_fields_and_frozen():
    status = AnnIndexStatus(
        exists=True,
        valid=True,
        ready=True,
        definition=_healthy_definition(),
        eligible_rows=12,
    )
    assert status.exists is True
    assert status.valid is True
    assert status.ready is True
    assert status.eligible_rows == 12
    assert "vector_cosine_ops" in (status.definition or "")
    with pytest.raises(Exception):
        status.exists = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Status catalog checks
# ---------------------------------------------------------------------------


async def test_status_missing_index():
    conn = _IndexConn(catalog=None, eligible_rows=3)
    status = await get_ann_index_status(conn, schema="noesis_core")
    assert status.exists is False
    assert status.valid is False
    assert status.ready is False
    assert status.definition is None
    assert status.eligible_rows == 3
    assert _status_is_healthy(status) is False


async def test_status_healthy_index():
    conn = _IndexConn(catalog=_catalog(), eligible_rows=12)
    status = await get_ann_index_status(conn, schema="noesis_core")
    assert status.exists is True
    assert status.valid is True
    assert status.ready is True
    assert status.eligible_rows == 12
    assert _status_is_healthy(status) is True


async def test_status_invalid_index_is_not_healthy():
    conn = _IndexConn(catalog=_catalog(valid=False), eligible_rows=12)
    status = await get_ann_index_status(conn, schema="noesis_core")
    assert status.exists is True
    assert status.valid is False
    assert _status_is_healthy(status) is False


async def test_status_not_ready_index_is_not_healthy():
    conn = _IndexConn(catalog=_catalog(ready=False), eligible_rows=12)
    status = await get_ann_index_status(conn, schema="noesis_core")
    assert status.exists is True
    assert status.ready is False
    assert _status_is_healthy(status) is False


async def test_status_same_name_wrong_definition_is_not_healthy():
    conn = _IndexConn(
        catalog=_catalog(definition=f"CREATE INDEX {INDEX_NAME} ON noesis_core.atoms USING btree (atom_id)"),
        eligible_rows=12,
    )
    status = await get_ann_index_status(conn, schema="noesis_core")
    assert status.exists is True
    assert "vector_cosine_ops" not in (status.definition or "")
    assert _status_is_healthy(status) is False


async def test_status_reports_eligible_row_count():
    conn = _IndexConn(catalog=None, eligible_rows=4242)
    status = await get_ann_index_status(conn, schema="noesis_core")
    assert status.eligible_rows == 4242


# ---------------------------------------------------------------------------
# Build / cold_start / force / conflict / lock
# ---------------------------------------------------------------------------


async def test_build_below_threshold_is_cold_start_noop():
    conn = _IndexConn(catalog=None, eligible_rows=THRESHOLD - 1)
    created = await build_ann_index(conn, schema="noesis_core")
    assert created is False
    assert conn.catalog is None
    assert not any("CREATE INDEX" in sql.upper() for sql in _sqls(conn))
    assert conn.lock_held is False


async def test_build_at_threshold_creates_index():
    conn = _IndexConn(catalog=None, eligible_rows=THRESHOLD)
    created = await build_ann_index(conn, schema="noesis_core")
    assert created is True
    assert any("CREATE INDEX" in sql.upper() and "CONCURRENTLY" in sql.upper() for sql in _sqls(conn))
    assert any("status = 'A'" in sql for sql in _sqls(conn))
    assert conn.analyzed is True
    assert conn.lock_held is False
    assert conn.transaction_used is False


async def test_build_force_bypasses_threshold():
    conn = _IndexConn(catalog=None, eligible_rows=10)
    created = await build_ann_index(conn, schema="noesis_core", force=True)
    assert created is True
    assert any("CREATE INDEX" in sql.upper() for sql in _sqls(conn))
    assert conn.lock_held is False


async def test_build_already_healthy_is_idempotent():
    conn = _IndexConn(catalog=_catalog(), eligible_rows=THRESHOLD)
    created = await build_ann_index(conn, schema="noesis_core")
    assert created is False
    assert not any("CREATE INDEX" in sql.upper() for sql in _sqls(conn))
    assert conn.lock_held is False


async def test_build_refuses_same_name_wrong_definition():
    conn = _IndexConn(
        catalog=_catalog(definition=f"CREATE INDEX {INDEX_NAME} ON noesis_core.atoms USING btree (atom_id)"),
        eligible_rows=THRESHOLD,
    )
    with pytest.raises(AnnIndexConflict):
        await build_ann_index(conn, schema="noesis_core")
    assert not any("CREATE INDEX" in sql.upper() for sql in _sqls(conn))
    assert conn.lock_held is False


async def test_build_refuses_invalid_or_unready_frozen_index():
    invalid = _IndexConn(catalog=_catalog(valid=False), eligible_rows=THRESHOLD)
    with pytest.raises(AnnIndexConflict):
        await build_ann_index(invalid, schema="noesis_core")
    assert not any("CREATE INDEX" in sql.upper() for sql in _sqls(invalid))
    assert invalid.lock_held is False

    unready = _IndexConn(catalog=_catalog(ready=False), eligible_rows=THRESHOLD)
    with pytest.raises(AnnIndexConflict):
        await build_ann_index(unready, schema="noesis_core", force=True)
    assert not any("CREATE INDEX" in sql.upper() for sql in _sqls(unready))
    assert unready.lock_held is False


async def test_build_raises_if_create_leaves_index_unhealthy():
    invalid = _IndexConn(catalog=None, eligible_rows=THRESHOLD, create_valid=False)
    with pytest.raises(AnnIndexError):
        await build_ann_index(invalid, schema="noesis_core")
    assert any("CREATE INDEX" in sql.upper() for sql in _sqls(invalid))
    assert invalid.lock_held is False

    unready = _IndexConn(catalog=None, eligible_rows=THRESHOLD, create_ready=False)
    with pytest.raises(AnnIndexError):
        await build_ann_index(unready, schema="noesis_core")
    assert unready.lock_held is False


async def test_build_refuses_when_advisory_lock_is_busy():
    conn = _IndexConn(catalog=None, eligible_rows=THRESHOLD, lock_available=False)
    with pytest.raises(AnnIndexBusy):
        await build_ann_index(conn, schema="noesis_core")
    assert not any("CREATE INDEX" in sql.upper() for sql in _sqls(conn))
    assert conn.lock_held is False


# ---------------------------------------------------------------------------
# Reindex / drop
# ---------------------------------------------------------------------------


async def test_reindex_missing_fails():
    conn = _IndexConn(catalog=None, eligible_rows=0)
    with pytest.raises(AnnIndexMissing):
        await reindex_ann_index(conn, schema="noesis_core")
    assert not any("CREATE INDEX" in sql.upper() for sql in _sqls(conn))
    assert conn.lock_held is False


async def test_reindex_existing_succeeds():
    conn = _IndexConn(catalog=_catalog(), eligible_rows=THRESHOLD)
    await reindex_ann_index(conn, schema="noesis_core")
    assert any("REINDEX" in sql.upper() and "CONCURRENTLY" in sql.upper() for sql in _sqls(conn))
    assert conn.analyzed is True
    assert conn.lock_held is False
    assert conn.transaction_used is False


async def test_drop_is_idempotent():
    conn = _IndexConn(catalog=_catalog(), eligible_rows=THRESHOLD)
    await drop_ann_index(conn, schema="noesis_core")
    assert conn.catalog is None
    await drop_ann_index(conn, schema="noesis_core")
    drops = [sql for sql in _sqls(conn) if "DROP INDEX" in sql.upper()]
    assert len(drops) >= 2
    assert all("CONCURRENTLY" in sql.upper() for sql in drops)
    assert conn.lock_held is False


# ---------------------------------------------------------------------------
# Task 7: CLI
# ---------------------------------------------------------------------------


def _cfg(schema: str = "noesis_core", url: str = "postgresql://u@h/noesis"):
    from types import SimpleNamespace

    return SimpleNamespace(noesis_schema=schema, noesis_database_url=url)


async def _run_cli(argv, conn, config=None):
    from hindsight_api.engine.retain import noesis_ann_index as index_mod

    async def connect(_config):
        return conn

    return await index_mod._main(argv, connect=connect, config=config or _cfg())


def test_cli_load_config_uses_hindsight_config(monkeypatch):
    import hindsight_api.config as config_module
    import hindsight_api.engine.retain.noesis_ann_index as index_mod

    sentinel = object()
    dotenv_loaded = False

    def load_dotenv():
        nonlocal dotenv_loaded
        dotenv_loaded = True

    monkeypatch.setattr(config_module, "load_dotenv_for_entrypoint", load_dotenv)
    monkeypatch.setattr(
        config_module.HindsightConfig,
        "from_env",
        classmethod(lambda cls: sentinel),
    )

    assert index_mod._load_config() is sentinel
    assert dotenv_loaded


async def test_cli_illegal_action_exits_2():
    assert await _run_cli(["explode"], _IndexConn()) == 2


async def test_cli_force_rejected_except_on_build():
    assert await _run_cli(["status", "--force"], _IndexConn()) == 2
    assert await _run_cli(["reindex", "--force"], _IndexConn()) == 2
    assert await _run_cli(["drop", "--force"], _IndexConn()) == 2


async def test_cli_illegal_schema_or_url_exits_2():
    assert await _run_cli(["status"], _IndexConn(), config=_cfg(schema="bad name")) == 2
    assert await _run_cli(["status"], _IndexConn(), config=_cfg(url="")) == 2


async def test_cli_status_and_build_success_exit_0():
    missing = _IndexConn(eligible_rows=10)
    assert await _run_cli(["status"], missing) == 0
    assert missing.closed is True
    created = _IndexConn(eligible_rows=THRESHOLD)
    assert await _run_cli(["build"], created) == 0
    assert created.closed is True
    forced = _IndexConn(eligible_rows=10)
    assert await _run_cli(["build", "--force"], forced) == 0


async def test_cli_unhealthy_or_conflict_exits_3():
    unhealthy = _IndexConn(catalog=_catalog(valid=False))
    assert await _run_cli(["status"], unhealthy) == 3
    conflict = _IndexConn(
        catalog=_catalog(definition=f"CREATE INDEX {INDEX_NAME} ON noesis_core.atoms USING btree (atom_id)"),
        eligible_rows=THRESHOLD,
    )
    assert await _run_cli(["build"], conflict) == 3
    assert conflict.closed is True
    assert conflict.lock_held is False
    invalid = _IndexConn(catalog=_catalog(valid=False), eligible_rows=THRESHOLD)
    assert await _run_cli(["build"], invalid) == 3
    assert invalid.closed is True
    unready = _IndexConn(catalog=_catalog(ready=False), eligible_rows=10)
    assert await _run_cli(["build", "--force"], unready) == 3
    assert not any("CREATE INDEX" in sql.upper() for sql in _sqls(unready))


async def test_cli_build_create_unhealthy_exits_4():
    invalid = _IndexConn(catalog=None, eligible_rows=THRESHOLD, create_valid=False)
    assert await _run_cli(["build"], invalid) == 4
    assert invalid.closed is True
    unready = _IndexConn(catalog=None, eligible_rows=10, create_ready=False)
    assert await _run_cli(["build", "--force"], unready) == 4
    assert unready.closed is True


async def test_cli_reindex_missing_exits_4_and_closes():
    conn = _IndexConn(catalog=None)
    assert await _run_cli(["reindex"], conn) == 4
    assert conn.closed is True
    assert conn.lock_held is False


async def test_cli_ddl_exception_exits_4_and_closes(monkeypatch):
    import hindsight_api.engine.retain.noesis_ann_index as index_mod

    conn = _IndexConn(eligible_rows=THRESHOLD)

    async def boom(*_args, **_kwargs):
        raise RuntimeError("ddl failed")

    monkeypatch.setattr(index_mod, "build_ann_index", boom)
    assert await _run_cli(["build"], conn) == 4
    assert conn.closed is True


def test_cli_epilog_names_embedding_rebuild_not_identity_rebuild():
    assert "noesis_embedding_rebuild" in _MODEL_SWITCH_EPILOG
    assert "identity_rebuild" not in _MODEL_SWITCH_EPILOG
    assert "REINDEX" in _MODEL_SWITCH_EPILOG
    assert "drop that index" in _MODEL_SWITCH_EPILOG


def test_schema_defers_frozen_ivfflat_and_points_at_embedding_rebuild():
    schema_sql = (
        Path(__file__).resolve().parents[3] / "docs" / "db" / "noesis-stage1-schema.sql"
    )
    text = schema_sql.read_text(encoding="utf-8")
    live_lines = [
        line for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("--")
    ]
    live = "\n".join(live_lines)
    assert "idx_atoms_embedding_ivfflat" not in live
    assert "ivfflat" not in live.lower()
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_atoms_embedding_ivfflat" in text
    assert "USING ivfflat (embedding vector_cosine_ops)" in text
    assert "WITH (lists = 100)" in text
    assert "status = 'A'" in text
    assert "atom_type IN ('E', 'P')" in text
    assert "embedding IS NOT NULL" in text
    assert "empty-database initialization" in text
    assert "cannot run inside a" in text and "transaction block" in text
    assert "noesis_ann_index build" in text
    assert "noesis_embedding_rebuild" in text
    assert "noesis_identity_rebuild" not in text
