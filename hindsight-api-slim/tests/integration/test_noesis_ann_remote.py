"""Remote PostgreSQL tests for ANN Top-100 and IVFFlat (requirement 04 Task 9).

Uses a unique temporary schema so ``noesis_core`` is never truncated or dropped.
Writes happen only inside that temp schema. Production ``noesis_core`` is
touched only by read-only ``status``. Gated: skipped unless the SSH env vars
from requirement 04 §16.3 are set. Credentials stay in the environment.

The normal offline regression never reaches the remote database.
"""

from __future__ import annotations

import logging
import random
import string
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

pytest.importorskip("paramiko")
pytest.importorskip("asyncpg")
pytest.importorskip("pgvector")

from hindsight_api.engine.retain.noesis_ann import (  # noqa: E402
    _ANN_QUERY_BY_TYPE,
    _IVFFLAT_LOCAL,
    AnnProfileUnavailable,
    _sql,
    recall_ann_candidates,
)
from hindsight_api.engine.retain.noesis_ann_index import (  # noqa: E402
    INDEX_NAME,
    build_ann_index,
    definition_is_frozen,
    drop_ann_index,
    get_ann_index_status,
    reindex_ann_index,
    status_is_healthy,
)
from tests.integration.noesis_remote_support import (  # noqa: E402
    SshTunnel as _SshTunnel,
)
from tests.integration.noesis_remote_support import (
    remote_enabled as _remote_enabled,
)

_DIM = 1024
_FILLER_ROWS = 120
_PROD_SCHEMA = "noesis_core"
_LOG = logging.getLogger(__name__)


requires_remote = pytest.mark.skipif(
    not _remote_enabled(),
    reason="NOESIS_REMOTE_TEST=1 and all NOESIS_SSH_* env vars are required",
)


def test_remote_gate_requires_explicit_opt_in(monkeypatch):
    for key in ("NOESIS_SSH_HOST", "NOESIS_SSH_PORT", "NOESIS_SSH_USER", "NOESIS_SSH_PW"):
        monkeypatch.setenv(key, "set")
    monkeypatch.delenv("NOESIS_REMOTE_TEST", raising=False)
    assert _remote_enabled() is False


def test_remote_gate_accepts_opt_in_with_all_ssh_values(monkeypatch):
    monkeypatch.setenv("NOESIS_REMOTE_TEST", "1")
    for key in ("NOESIS_SSH_HOST", "NOESIS_SSH_PORT", "NOESIS_SSH_USER", "NOESIS_SSH_PW"):
        monkeypatch.setenv(key, "set")
    assert _remote_enabled() is True


def _axis_vector(index: int) -> list[float]:
    vector = [0.0] * _DIM
    vector[index % _DIM] = 1.0
    return vector


def _blend(left: int, right: int, left_weight: float) -> list[float]:
    vector = [0.0] * _DIM
    vector[left % _DIM] = left_weight
    vector[right % _DIM] = 1.0 - left_weight
    return vector


async def _connect():
    import asyncpg
    from pgvector.asyncpg import register_vector

    tunnel = _SshTunnel()
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=tunnel.port,
        user="postgres",
        database="noesis",
        timeout=20,
    )
    await register_vector(conn)
    return tunnel, conn


async def _create_minimal_tables(conn, schema: str) -> None:
    await conn.execute(f"CREATE SCHEMA {schema}")
    await conn.execute(
        f"""
        CREATE TABLE {schema}.atoms (
            atom_id BIGSERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            atom_type TEXT NOT NULL,
            embedding VECTOR({_DIM}),
            status TEXT NOT NULL
        )
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE {schema}.embedding_profiles (
            embedding_kind VARCHAR(32) PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_revision TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            status VARCHAR(20) NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    await conn.execute(
        f"""
        INSERT INTO {schema}.embedding_profiles
            (embedding_kind, model_name, model_revision, dimension, status)
        VALUES ('identity', 'bge-m3', 'bge-m3-1024-v1', {_DIM}, 'ready')
        """
    )


async def _insert_fixture(conn, schema: str) -> dict[str, int]:
    rows = [
        ("苹果手机", "E", "A", _blend(0, 1, 0.95)),
        ("iPhone", "E", "A", _blend(0, 1, 0.90)),
        ("智能手机", "E", "A", _blend(0, 2, 0.85)),
        ("航空母舰", "E", "A", _axis_vector(500)),
        ("买", "P", "A", _axis_vector(10)),
        ("卖掉", "P", "A", _blend(10, 11, 0.9)),
        ("周末计划", "G", "A", _axis_vector(0)),
        ("旧手机", "E", "D", _axis_vector(0)),
        ("空向量实体", "E", "A", None),
    ]
    ids: dict[str, int] = {}
    for text, atom_type, status, embedding in rows:
        atom_id = await conn.fetchval(
            f"INSERT INTO {schema}.atoms (text, atom_type, status, embedding) "
            f"VALUES ($1, $2, $3, $4) RETURNING atom_id",
            text,
            atom_type,
            status,
            embedding,
        )
        ids[text] = int(atom_id)
    for index in range(_FILLER_ROWS):
        await conn.execute(
            f"INSERT INTO {schema}.atoms (text, atom_type, status, embedding) "
            f"VALUES ($1, $2, $3, $4)",
            f"filler_{index}",
            "E",
            "A",
            _axis_vector(20 + (index % 800)),
        )
    return ids


@requires_remote
async def test_ann_temp_schema_recall_ivfflat_and_cleanup():
    tunnel, conn = await _connect()
    schema = f"n04ann_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"
    leftover = None
    try:
        await _create_minimal_tables(conn, schema)
        _LOG.info("ann remote temp schema created=%s", schema)
        ids = await _insert_fixture(conn, schema)

        before = await get_ann_index_status(conn, schema=schema)
        assert before.exists is False

        e_hits = await recall_ann_candidates(
            conn, schema=schema, source_atom_id=ids["苹果手机"], limit=10
        )
        e_texts = [row.text for row in e_hits]
        e_types = {row.atom_type for row in e_hits}
        assert e_types <= {"E"}
        assert "苹果手机" not in e_texts
        assert "买" not in e_texts
        assert "卖掉" not in e_texts
        assert "周末计划" not in e_texts
        assert "旧手机" not in e_texts
        assert "空向量实体" not in e_texts
        assert "iPhone" in e_texts
        assert all(row.similarity == pytest.approx(1.0 - row.distance) for row in e_hits)

        p_hits = await recall_ann_candidates(
            conn, schema=schema, source_atom_id=ids["买"], limit=10
        )
        p_texts = [row.text for row in p_hits]
        assert {row.atom_type for row in p_hits} <= {"P"}
        assert "买" not in p_texts
        assert "卖掉" in p_texts
        assert "苹果手机" not in p_texts

        await conn.execute(
            f"UPDATE {schema}.embedding_profiles SET status = 'rebuilding' "
            f"WHERE embedding_kind = 'identity'"
        )
        with pytest.raises(AnnProfileUnavailable):
            await recall_ann_candidates(
                conn, schema=schema, source_atom_id=ids["苹果手机"], limit=10
            )
        await conn.execute(
            f"UPDATE {schema}.embedding_profiles SET status = 'ready' "
            f"WHERE embedding_kind = 'identity'"
        )

        created = await build_ann_index(conn, schema=schema, force=True)
        assert created is True
        status = await get_ann_index_status(conn, schema=schema)
        _LOG.info(
            "ann remote catalog exists=%s valid=%s ready=%s eligible_rows=%s def=%s",
            status.exists,
            status.valid,
            status.ready,
            status.eligible_rows,
            status.definition,
        )
        assert status_is_healthy(status, schema=schema)
        assert status.valid is True
        assert status.ready is True
        assert definition_is_frozen(status.definition, schema=schema)
        assert status.definition is not None
        lowered = status.definition.lower()
        assert "ivfflat" in lowered
        assert "vector_cosine_ops" in status.definition
        assert "lists='100'" in status.definition or "lists = 100" in status.definition
        assert "status = 'A'" in status.definition
        assert "atom_type" in status.definition
        assert "embedding IS NOT NULL" in status.definition

        # Production recall SQL (MATERIALIZED CTE), not a simplified substitute.
        query = _sql(schema, _ANN_QUERY_BY_TYPE["E"])
        explain_sql = "EXPLAIN\n" + query
        assert "WITH source AS MATERIALIZED" in query
        assert "CROSS JOIN" not in query
        assert "ORDER BY a.embedding <=> (SELECT embedding FROM source)" in query
        assert "1.0 - distance AS similarity" in query
        assert "status = 'A'" in query
        assert "atom_type = 'E'" in query
        async with conn.transaction(readonly=True):
            await conn.execute("SET LOCAL enable_seqscan = off")
            for statement in _IVFFLAT_LOCAL:
                await conn.execute(statement)
            plan_rows = await conn.fetch(explain_sql, ids["苹果手机"], 10)
        plan = "\n".join(row[0] for row in plan_rows)
        _LOG.info("ann remote explain=\n%s", plan)
        # The production query shape must expose a KNN execution parameter so
        # PostgreSQL can select the frozen IVFFlat index.
        assert INDEX_NAME in plan
        assert "a.embedding <=> " in plan or "embedding <=> " in plan

        await reindex_ann_index(conn, schema=schema)
        reindexed = await get_ann_index_status(conn, schema=schema)
        assert status_is_healthy(reindexed, schema=schema)

        await drop_ann_index(conn, schema=schema)
        dropped = await get_ann_index_status(conn, schema=schema)
        _LOG.info("ann remote after drop exists=%s", dropped.exists)
        assert dropped.exists is False
    finally:
        try:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            leftover = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.schemata "
                "WHERE schema_name = $1)",
                schema,
            )
            _LOG.info("ann remote leftover=%s schema=%s", leftover, schema)
        finally:
            await conn.close()
            tunnel.close()
    assert leftover is False


@requires_remote
async def test_ann_production_status_is_readonly_and_not_forced_on_empty():
    tunnel, conn = await _connect()
    try:
        extensions = await conn.fetch(
            "SELECT extname, extversion FROM pg_extension "
            "ORDER BY extname"
        )
        _LOG.info(
            "ann production extensions=%s",
            [(row["extname"], row["extversion"]) for row in extensions],
        )
        status = await get_ann_index_status(conn, schema=_PROD_SCHEMA)
        _LOG.info(
            "ann production status exists=%s valid=%s ready=%s eligible_rows=%s",
            status.exists,
            status.valid,
            status.ready,
            status.eligible_rows,
        )
        if status.eligible_rows == 0:
            assert status.exists is False
        assert status.eligible_rows >= 0
    finally:
        await conn.close()
        tunnel.close()
