"""Remote PostgreSQL tests for ANN Top-100 and IVFFlat (requirement 04 Task 9).

Uses a unique temporary schema so ``noesis_core`` is never truncated or dropped.
Writes happen only inside that temp schema. Production ``noesis_core`` is
touched only by read-only ``status``. Gated: skipped unless the SSH env vars
from requirement 04 §16.3 are set. Credentials stay in the environment.

The normal offline regression never reaches the remote database.
"""

from __future__ import annotations

import logging
import os
import random
import socket
import string
import threading
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

pytest.importorskip("paramiko")
pytest.importorskip("asyncpg")
pytest.importorskip("pgvector")

from hindsight_api.engine.retain.noesis_ann import (  # noqa: E402
    _IVFFLAT_LOCAL,
    AnnProfileUnavailable,
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

_EXPECTED_HOST_KEY = "SHA256:fYPgM4a2OY1ZRhdQbx2z2YjiQ9bOMx4zo/c1ewn+WCs"
_DIM = 1024
_FILLER_ROWS = 120
_PROD_SCHEMA = "noesis_core"
_LOG = logging.getLogger(__name__)


def _remote_enabled() -> bool:
    return all(
        os.environ.get(key)
        for key in ("NOESIS_SSH_HOST", "NOESIS_SSH_PORT", "NOESIS_SSH_USER", "NOESIS_SSH_PW")
    )


requires_remote = pytest.mark.skipif(not _remote_enabled(), reason="NOESIS_SSH_* env not set")


class _SshTunnel:
    """Minimal paramiko direct-tcpip forwarder (one local port, N sockets)."""

    def __init__(self) -> None:
        import base64
        import hashlib

        import paramiko

        self.transport = paramiko.Transport(
            (os.environ["NOESIS_SSH_HOST"], int(os.environ["NOESIS_SSH_PORT"]))
        )
        self.transport.start_client(timeout=20)
        key = self.transport.get_remote_server_key()
        fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        if fingerprint != _EXPECTED_HOST_KEY:
            self.transport.close()
            raise RuntimeError("remote host key mismatch — refusing to connect")
        self.transport.auth_password(
            username=os.environ["NOESIS_SSH_USER"],
            password=os.environ["NOESIS_SSH_PW"],
        )
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(8)
        self.port = self._server.getsockname()[1]
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stopping.is_set():
            try:
                client, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._forward, args=(client,), daemon=True).start()

    def _forward(self, client: socket.socket) -> None:
        try:
            channel = self.transport.open_channel(
                "direct-tcpip", ("localhost", 5432), client.getsockname()
            )
        except Exception:
            client.close()
            return

        def pump(src, dst):
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                try:
                    dst.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        threading.Thread(target=pump, args=(client, channel), daemon=True).start()
        pump(channel, client)

    def close(self) -> None:
        self._stopping.set()
        try:
            self._server.close()
        except OSError:
            pass
        self.transport.close()


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
        ("苹果手机", "E", "active", _blend(0, 1, 0.95)),
        ("iPhone", "E", "active", _blend(0, 1, 0.90)),
        ("智能手机", "E", "active", _blend(0, 2, 0.85)),
        ("航空母舰", "E", "active", _axis_vector(500)),
        ("买", "P", "active", _axis_vector(10)),
        ("卖掉", "P", "active", _blend(10, 11, 0.9)),
        ("周末计划", "G", "active", _axis_vector(0)),
        ("旧手机", "E", "inactive", _axis_vector(0)),
        ("空向量实体", "E", "active", None),
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
            "active",
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
        assert "active" in status.definition
        assert "atom_type" in status.definition
        assert "embedding IS NOT NULL" in status.definition

        explain_sql = f"""
EXPLAIN
SELECT a.atom_id
FROM {schema}.atoms AS a
WHERE a.status = 'active'
  AND a.atom_type = 'E'
  AND a.embedding IS NOT NULL
  AND a.atom_id <> $1
ORDER BY a.embedding <=> (
    SELECT s.embedding FROM {schema}.atoms AS s WHERE s.atom_id = $1
)
LIMIT $2
"""
        async with conn.transaction(readonly=True):
            await conn.execute("SET LOCAL enable_seqscan = off")
            for statement in _IVFFLAT_LOCAL:
                await conn.execute(statement)
            plan_rows = await conn.fetch(explain_sql, ids["苹果手机"], 10)
        plan = "\n".join(row[0] for row in plan_rows)
        _LOG.info("ann remote explain=\n%s", plan)
        assert INDEX_NAME in plan

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
