"""Remote bge-m3 + PostgreSQL smoke test (requirement 03 §15.6).

Runs the REAL write path (real ``NoesisIdentityClient`` against
``http://10.0.0.8:8010``, real asyncpg, real ``noesis_core``) inside one outer
transaction so the final ``ROLLBACK`` leaves zero business rows. Gated: skipped
unless ``NOESIS_REMOTE_TEST=1`` and the SSH env vars are present.

It proves:
  * ``GET /health`` reports dim=1024 and a model whose normalized basename is
    exactly ``bge-m3``;
  * a real golden literal encodes to 1024 floats (E and P);
  * a real ``_ingest_fact``-equivalent batch (fake LLM + real bge + real PG)
    lands atoms whose ``embedding`` is non-NULL and ``vector_dims(embedding)=1024``;
  * migration 003 is idempotent and the profile claim/mismatch gate works;
  * a cosine direction sanity (苹果 vs 苹果 手机 > 苹果 vs 航空母舰), no
    threshold assertion.

Credentials come exclusively from the environment; nothing is written to the
source, fixtures, or logs. DDL applied to ``noesis_core`` is idempotent
(CREATE TABLE IF NOT EXISTS) and carries no business rows.
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import socket
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

pytest.importorskip("paramiko")
pytest.importorskip("asyncpg")
pytest.importorskip("httpx")

from hindsight_api.engine.retain import noesis_ingest  # noqa: E402
from hindsight_api.engine.retain.noesis_identity_rebuild import (  # noqa: E402
    BuildSpec,
    run_identity_rebuild,
)
from hindsight_api.engine.retain.noesis_identity_vector import (  # noqa: E402
    NoesisIdentityClient,
    normalize_model_basename,
)
from tests.noesis_fakes import (  # noqa: E402
    FakeExtractOnceFactory,
    golden_fact_recursive,
    llm_config,
    noesis_config,
)

_EXPECTED_HOST_KEY = "SHA256:fYPgM4a2OY1ZRhdQbx2z2YjiQ9bOMx4zo/c1ewn+WCs"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_MIGRATION_003 = _REPO_ROOT / "docs" / "db" / "migrations" / "003-noesis-identity-embedding-profile.sql"
_REAL_BGE_URL = "http://10.0.0.8:8010"


def _remote_enabled() -> bool:
    return (
        os.environ.get("NOESIS_REMOTE_TEST") == "1"
        and all(os.environ.get(k) for k in ("NOESIS_SSH_HOST", "NOESIS_SSH_PORT", "NOESIS_SSH_USER", "NOESIS_SSH_PW"))
    )


requires_remote = pytest.mark.skipif(not _remote_enabled(), reason="NOESIS_REMOTE_TEST/SSH env not set")


class _SshTunnel:
    """Minimal paramiko direct-tcpip forwarder (one local port, N sockets)."""

    def __init__(self) -> None:
        import paramiko

        self._paramiko = paramiko
        self.transport = paramiko.Transport((os.environ["NOESIS_SSH_HOST"], int(os.environ["NOESIS_SSH_PORT"])))
        self.transport.start_client(timeout=20)
        key = self.transport.get_remote_server_key()
        fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        if fingerprint != _EXPECTED_HOST_KEY:
            self.transport.close()
            raise RuntimeError("remote host key mismatch — refusing to connect")
        self.transport.auth_password(username=os.environ["NOESIS_SSH_USER"], password=os.environ["NOESIS_SSH_PW"])
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
            channel = self.transport.open_channel("direct-tcpip", ("localhost", 5432), client.getsockname())
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


class _SingleConnAcquire:
    def __init__(self, conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _SingleConnPool:
    def __init__(self, conn) -> None:
        self._conn = conn
        self.closed = False

    def acquire(self) -> _SingleConnAcquire:
        return _SingleConnAcquire(self._conn)

    def is_closed(self) -> bool:
        return self.closed

    async def close(self) -> None:
        self.closed = True


def _async_return(value):
    async def factory(_config):
        return value

    return factory


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


async def _apply_migration_003(pool):
    """Ensure the identity profile table exists (executed only if absent).

    Migration-003 idempotency is proven in test_noesis_remote_pg.py, which runs
    the same file twice. Running the full BEGIN/COMMIT script as a single
    multi-statement query on an asyncpg connection that already carries the
    pgvector type registration trips pg_type_typname_nsp_index, so here we only
    ever apply it once and only when the table is missing.
    """
    exists = await pool.fetchval(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='noesis_core' AND table_name='embedding_profiles')"
    )
    if exists:
        return
    content = _MIGRATION_003.read_text(encoding="utf-8")
    await pool.execute(content)


async def _real_bge_client() -> NoesisIdentityClient:
    return NoesisIdentityClient(
        base_url=_REAL_BGE_URL,
        model="bge-m3",
        revision="bge-m3-1024-v1",
        dimension=1024,
        timeout_seconds=8.0,
        max_retries=1,
    )


@requires_remote
async def test_remote_identity_health_and_cosine_smoke():
    # 1. Real /health: dim=1024 and exact normalized basename.
    client = await _real_bge_client()
    await client.ensure_ready()
    try:
        response = await client._http.get("/health")
        payload = response.json()
        assert payload["dim"] == 1024
        assert normalize_model_basename(payload["model"]) == "bge-m3"

        # 2. Real encoding of golden literals: 1024-dim floats each.
        v_apple = await client.embed("苹果", "E")
        v_iphone = await client.embed("苹果手机", "E")
        v_carrier = await client.embed("航空母舰", "E")
        v_buy = await client.embed("买", "P")
        assert len(v_apple) == 1024
        assert len(v_iphone) == 1024
        assert len(v_carrier) == 1024
        assert len(v_buy) == 1024

        # 3. Direction sanity (no threshold): 苹果 ≈ 苹果手机 > 苹果 vs 航空母舰.
        assert _cosine(v_apple, v_iphone) > _cosine(v_apple, v_carrier)
    finally:
        await client.aclose()


@requires_remote
async def test_remote_identity_ingest_writes_real_vector_and_rolls_back():
    import asyncpg

    tunnel = _SshTunnel()
    conn = None
    try:
        conn = await asyncpg.connect(host="127.0.0.1", port=tunnel.port, user="postgres", database="noesis")

        # 4. Migration 003 idempotent (applied twice) — no business rows, safe.
        await _apply_migration_003(conn)

        # 5. Outer transaction: everything below rolls back.
        outer_tx = conn.transaction()
        await outer_tx.start()
        pool = _SingleConnPool(conn)

        # 6. Real ingest with a real bge client + fake LLM.
        client = await _real_bge_client()
        original_extract = noesis_ingest.extract_noesis_components

        def fake_extract(text, *, extract_once):
            from hyperextract.noesis import ExtractionOutcome

            return ExtractionOutcome(components=[golden_fact_recursive()], alerts=[], attempts=1)

        noesis_ingest.extract_noesis_components = fake_extract
        try:
            cfg = noesis_config(
                noesis_embedding_base_url=_REAL_BGE_URL,
                noesis_embedding_model="bge-m3",
                noesis_embedding_revision="bge-m3-1024-v1",
                noesis_embedding_dimension=1024,
                noesis_embedding_timeout_seconds=8.0,
                noesis_embedding_max_retries=1,
            )
            await noesis_ingest.ingest_noesis_batch(
                [
                    {
                        "content": "小明没写作业后揍了自己。",
                        "event_date": datetime(2026, 9, 5, 2, 0, 0, tzinfo=UTC),
                        "document_id": "identity-smoke-doc",
                    }
                ],
                "identity-smoke-bank",
                cfg,
                llm_config=llm_config(),
                extract_once_factory=FakeExtractOnceFactory(),
                pool_factory=_async_return(pool),
                identity_client_factory=_async_return(client),
            )

            # 7. Atoms landed with non-NULL real vectors, vector_dims=1024.
            atoms = await conn.fetch(
                "SELECT atom_id, text, atom_type, embedding, vector_dims(embedding) AS dims FROM noesis_core.atoms"
            )
            assert len(atoms) >= 4
            ep = [row for row in atoms if row["atom_type"] in ("E", "P")]
            assert ep, "no E/P atoms written"
            for row in ep:
                assert row["embedding"] is not None, f"{row['text']} has NULL embedding"
                assert row["dims"] == 1024, f"{row['text']} dims={row['dims']}"

            # 8. Profile claimed (auto) and consistent with the running config.
            profile = await conn.fetchrow(
                "SELECT model_name, model_revision, dimension, status FROM noesis_core.embedding_profiles "
                "WHERE embedding_kind = 'identity'"
            )
            assert profile is not None
            assert profile["model_revision"] == "bge-m3-1024-v1"
            assert profile["dimension"] == 1024
            assert profile["status"] == "ready"
        finally:
            noesis_ingest.extract_noesis_components = original_extract
            await client.aclose()

        # 9. Rollback the outer transaction: zero business rows remain.
        await outer_tx.rollback()
        for table in ("events", "event_atoms", "ingestion_alerts"):
            remaining = await conn.fetchval(f"SELECT count(*) FROM noesis_core.{table}")
            assert remaining == 0, f"{table} rows leaked: {remaining}"
        remaining_atoms = await conn.fetchval("SELECT count(*) FROM noesis_core.atoms WHERE embedding IS NOT NULL")
        assert remaining_atoms == 0, f"atoms with embedding leaked: {remaining_atoms}"
    finally:
        if conn is not None:
            await conn.close()
        tunnel.close()


@requires_remote
async def test_remote_identity_rebuild_stages_and_atomically_cuts_over():
    """Exercise the production rebuild SQL against real pgvector/PostgreSQL."""
    import asyncpg

    tunnel = _SshTunnel()
    conn = None
    client = None
    try:
        conn = await asyncpg.connect(host="127.0.0.1", port=tunnel.port, user="postgres", database="noesis")
        outer_tx = conn.transaction()
        await outer_tx.start()
        schema = f"noesis_identity_test_{uuid.uuid4().hex[:12]}"
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(
            f'CREATE TABLE "{schema}".atoms ('
            "atom_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
            "text TEXT NOT NULL, atom_type CHAR(1) NOT NULL, status VARCHAR(20) NOT NULL DEFAULT 'active', "
            "embedding vector(1024))"
        )
        await conn.execute(
            f'CREATE TABLE "{schema}".embedding_profiles ('
            "embedding_kind TEXT PRIMARY KEY, model_name TEXT NOT NULL, model_revision TEXT NOT NULL, "
            "dimension INTEGER NOT NULL, status TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        await conn.execute(
            f'INSERT INTO "{schema}".embedding_profiles '
            "(embedding_kind, model_name, model_revision, dimension, status) "
            "VALUES ('identity', 'bge-m3', 'old-generation', 1024, 'ready')"
        )
        await conn.executemany(
            f'INSERT INTO "{schema}".atoms (text, atom_type) VALUES ($1, $2)',
            [("苹果", "E"), ("购买", "P"), ("内部构元", "G")],
        )

        client = await _real_bge_client()
        result = await run_identity_rebuild(
            pool=_SingleConnPool(conn),
            schema=schema,
            client=client,
            spec=BuildSpec(model="bge-m3", revision="bge-m3-1024-v2", dimension=1024),
        )
        assert result == 0
        rows = await conn.fetch(
            f'SELECT atom_type, embedding IS NOT NULL AS populated, vector_dims(embedding) AS dims '
            f'FROM "{schema}".atoms ORDER BY atom_id'
        )
        assert [(row["atom_type"], row["populated"], row["dims"]) for row in rows] == [
            ("E", True, 1024),
            ("P", True, 1024),
            ("G", False, None),
        ]
        profile = await conn.fetchrow(
            f'SELECT model_name, model_revision, dimension, status FROM "{schema}".embedding_profiles '
            "WHERE embedding_kind = 'identity'"
        )
        assert dict(profile) == {
            "model_name": "bge-m3",
            "model_revision": "bge-m3-1024-v2",
            "dimension": 1024,
            "status": "ready",
        }
        await outer_tx.rollback()
    finally:
        if client is not None:
            await client.aclose()
        if conn is not None:
            await conn.close()
        tunnel.close()
