"""Remote bge-m3 + PostgreSQL smoke test (requirement 03 §15.6).

Runs the REAL write path (real ``NoesisEmbeddingClient`` against the bge-m3
service reached through the same SSH tunnel as ``noesis_core`` — the 3138 host
runs an identical ``semnorm`` service on ``localhost:8010``, with the legacy
direct ``http://10.0.0.8:8010`` kept only as the default fallback, real asyncpg,
real ``noesis_core``) inside one outer transaction so the final ``ROLLBACK``
leaves zero business rows. Gated: skipped unless ``NOESIS_REMOTE_TEST=1`` and
the SSH env vars are present.

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
import json
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
from hindsight_api.engine.retain.noesis_embedding import (  # noqa: E402
    NoesisEmbeddingClient,
    normalize_model_basename,
)
from hindsight_api.engine.retain.noesis_embedding_rebuild import (  # noqa: E402
    BuildSpec,
    run_embedding_rebuild,
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

# Serialize with every other remote noesis_core test under pytest-xdist (see
# test_noesis_remote_pg.remote_group for the why).
remote_group = pytest.mark.xdist_group("noesis_remote")


class _SshTunnel:
    """Minimal paramiko direct-tcpip forwarder (N local ports, M sockets each).

    ``remote_ports`` are forwarded from a bound local port to
    ``localhost:<remote_port>`` on the SSH host; ``ports[remote_port]`` is the
    matching local port. ``port`` stays the PostgreSQL alias for compatibility.
    """

    def __init__(self, remote_ports: tuple[int, ...] = (5432,)) -> None:
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
        self._stopping = threading.Event()
        self._servers: list[socket.socket] = []
        self.ports: dict[int, int] = {}
        for remote_port in remote_ports:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen(8)
            self._servers.append(server)
            self.ports[remote_port] = server.getsockname()[1]
            threading.Thread(target=self._serve, args=(server, remote_port), daemon=True).start()
        self.port = self.ports.get(5432, next(iter(self.ports.values())))

    def _serve(self, server: socket.socket, remote_port: int) -> None:
        while not self._stopping.is_set():
            try:
                client, _ = server.accept()
            except OSError:
                return
            threading.Thread(target=self._forward, args=(client, remote_port), daemon=True).start()

    def _forward(self, client: socket.socket, remote_port: int) -> None:
        try:
            channel = self.transport.open_channel("direct-tcpip", ("localhost", remote_port), client.getsockname())
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
        for server in self._servers:
            try:
                server.close()
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


async def _real_bge_client(base_url: str = _REAL_BGE_URL) -> NoesisEmbeddingClient:
    return NoesisEmbeddingClient(
        base_url=base_url,
        model="bge-m3",
        revision="bge-m3-1024-v1",
        dimension=1024,
        timeout_seconds=8.0,
        max_retries=1,
    )


@requires_remote
@remote_group
async def test_remote_identity_health_and_cosine_smoke():
    # 1. Real /health: dim=1024 and exact normalized basename.
    tunnel = _SshTunnel(remote_ports=(8010,))
    bge_url = f"http://127.0.0.1:{tunnel.ports[8010]}"
    client = await _real_bge_client(bge_url)
    try:
        await client.ensure_ready()
        response = await client._http.get("/health")
        payload = response.json()
        assert payload["dim"] == 1024
        assert normalize_model_basename(payload["model"]) == "bge-m3"

        # 2. Real encoding of golden literals: 1024-dim floats each.
        v_apple = await client.embed_identity("苹果", "E")
        v_iphone = await client.embed_identity("苹果手机", "E")
        v_carrier = await client.embed_identity("航空母舰", "E")
        v_buy = await client.embed_identity("买", "P")
        assert len(v_apple) == 1024
        assert len(v_iphone) == 1024
        assert len(v_carrier) == 1024
        assert len(v_buy) == 1024

        # 3. Direction sanity (no threshold): 苹果 ≈ 苹果手机 > 苹果 vs 航空母舰.
        assert _cosine(v_apple, v_iphone) > _cosine(v_apple, v_carrier)
    finally:
        await client.aclose()
        tunnel.close()


@requires_remote
@remote_group
async def test_remote_identity_ingest_writes_real_vector_and_rolls_back():
    import asyncpg

    tunnel = _SshTunnel(remote_ports=(5432, 8010))
    bge_url = f"http://127.0.0.1:{tunnel.ports[8010]}"
    conn = None
    try:
        conn = await asyncpg.connect(host="127.0.0.1", port=tunnel.ports[5432], user="postgres", database="noesis")

        # 4. Migration 003 idempotent (applied twice) — no business rows, safe.
        await _apply_migration_003(conn)

        # 5. Outer transaction: everything below rolls back.
        outer_tx = conn.transaction()
        await outer_tx.start()
        pool = _SingleConnPool(conn)

        # 6. Real ingest with a real bge client + fake LLM.
        client = await _real_bge_client(bge_url)
        original_extract = noesis_ingest.extract_noesis_components

        def fake_extract(text, *, extract_once):
            from hyperextract.noesis import ExtractionOutcome

            return ExtractionOutcome(components=[golden_fact_recursive()], alerts=[], attempts=1)

        noesis_ingest.extract_noesis_components = fake_extract
        try:
            cfg = noesis_config(
                noesis_embedding_base_url=bge_url,
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
                embedding_client_factory=_async_return(client),
            )

            # 7. Atoms landed with non-NULL real vectors, vector_dims=1024.
            atoms = await conn.fetch(
                "SELECT atom_id, text, atom_type, embedding, vector_dims(embedding) AS dims FROM noesis_core.atoms"
            )
            assert len(atoms) >= 4
            # asyncpg decodes the 04A one-byte "char" enum as bytes.
            ep = [row for row in atoms if row["atom_type"] in (b"E", b"P")]
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
        for table in ("events", "event_atoms", "ingestion_alerts", "anchors"):
            remaining = await conn.fetchval(f"SELECT count(*) FROM noesis_core.{table}")
            assert remaining == 0, f"{table} rows leaked: {remaining}"
        remaining_atoms = await conn.fetchval("SELECT count(*) FROM noesis_core.atoms WHERE embedding IS NOT NULL")
        assert remaining_atoms == 0, f"atoms with embedding leaked: {remaining_atoms}"
    finally:
        if conn is not None:
            await conn.close()
        tunnel.close()


@requires_remote
@remote_group
async def test_remote_embedding_rebuild_stages_and_atomically_cuts_over():
    """Exercise the production rebuild SQL against real pgvector/PostgreSQL."""
    import asyncpg

    tunnel = _SshTunnel(remote_ports=(5432, 8010))
    bge_url = f"http://127.0.0.1:{tunnel.ports[8010]}"
    conn = None
    client = None
    try:
        conn = await asyncpg.connect(host="127.0.0.1", port=tunnel.ports[5432], user="postgres", database="noesis")
        outer_tx = conn.transaction()
        await outer_tx.start()
        schema = f"noesis_embedding_test_{uuid.uuid4().hex[:12]}"
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(
            f'CREATE TABLE "{schema}".atoms ('
            "atom_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
            "text TEXT NOT NULL, atom_type CHAR(1) NOT NULL, status VARCHAR(20) NOT NULL DEFAULT 'active', "
            "embedding vector(1024))"
        )
        await conn.execute(
            f'CREATE TABLE "{schema}".anchors ('
            "anchor_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
            f'atom_id BIGINT NOT NULL REFERENCES "{schema}".atoms(atom_id), '
            "centroid_vector vector(1024) NOT NULL, total_count BIGINT NOT NULL DEFAULT 0, "
            "status CHAR(1) NOT NULL DEFAULT 'A', updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        await conn.execute(f'CREATE TABLE "{schema}".events (event_id BIGINT PRIMARY KEY, data JSONB NOT NULL)')
        await conn.execute(
            f'CREATE TABLE "{schema}".event_atoms ('
            "event_id BIGINT NOT NULL, occurrence_id INT NOT NULL, atom_id BIGINT NOT NULL, "
            "anchor_id BIGINT NOT NULL, role_type CHAR(1) NOT NULL, target_occ INT, "
            "PRIMARY KEY (event_id, occurrence_id))"
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
        atom_ids: dict[str, int] = {}
        for text, atom_type in (("张三", "E"), ("修复", "P"), ("服务器", "E"), ("内部构元", "G")):
            row = await conn.fetchrow(
                f'INSERT INTO "{schema}".atoms (text, atom_type) VALUES ($1, $2) RETURNING atom_id',
                text,
                atom_type,
            )
            atom_ids[text] = row["atom_id"]
        zeros = "[" + ",".join(["0"] * 1024) + "]"
        anchor_ids: dict[str, int] = {}
        for text in ("张三", "修复", "服务器"):
            row = await conn.fetchrow(
                f'INSERT INTO "{schema}".anchors (atom_id, centroid_vector, total_count, status) '
                "VALUES ($1, $2::vector, 7, 'A') RETURNING anchor_id",
                atom_ids[text],
                zeros,
            )
            anchor_ids[text] = row["anchor_id"]
        component = {
            "utterance_type": "fact",
            "atoms": [
                {"pos": 1, "text": "张三", "type": "E", "role": "agent", "target_occ": 2, "resolved": None},
                {"pos": 2, "text": "修复", "type": "P", "role": "predicate", "target_occ": None, "resolved": None},
                {"pos": 3, "text": "服务器", "type": "E", "role": "patient", "target_occ": 2, "resolved": None},
            ],
            "tree": {
                "predicate": "修复",
                "agent": [{"text": "张三", "modifier": [], "implied": False}],
                "patient": [{"text": "服务器", "modifier": [], "implied": False}],
                "modifier": [],
                "nested": [],
                "conditional": [],
            },
        }
        await conn.execute(
            f'INSERT INTO "{schema}".events (event_id, data) VALUES (1, $1::jsonb)',
            json.dumps({"component": component}, ensure_ascii=False),
        )
        for occurrence_id, text, role, target, anchor_text in (
            (1, "张三", "A", 2, "张三"),
            (2, "修复", "R", None, "修复"),
            (3, "服务器", "P", 2, "服务器"),
        ):
            await conn.execute(
                f'INSERT INTO "{schema}".event_atoms '
                "(event_id, occurrence_id, atom_id, anchor_id, role_type, target_occ) "
                "VALUES (1, $1, $2, $3, $4, $5)",
                occurrence_id,
                atom_ids[text],
                anchor_ids[anchor_text],
                role,
                target,
            )

        client = await _real_bge_client(bge_url)
        result = await run_embedding_rebuild(
            pool=_SingleConnPool(conn),
            schema=schema,
            client=client,
            spec=BuildSpec(model="bge-m3", revision="bge-m3-1024-v2", dimension=1024),
        )
        assert result == 0

        # Every active E/P atom was rebuilt to the target generation; G is untouched.
        rows = await conn.fetch(
            f'SELECT text, embedding IS NOT NULL AS populated, vector_dims(embedding) AS dims '
            f'FROM "{schema}".atoms ORDER BY atom_id'
        )
        by_text = {row["text"]: row for row in rows}
        for text in ("张三", "修复", "服务器"):
            assert by_text[text]["populated"] is True, f"{text} has NULL embedding"
            assert by_text[text]["dims"] == 1024, f"{text} dims={by_text[text]['dims']}"
        assert by_text["内部构元"]["populated"] is False, "G must never get a vector"

        # E and P anchors were rebuilt from history: one sample each -> count 1, 1024 dims.
        anchor_rows = await conn.fetch(
            f'SELECT anchor_id, total_count, vector_dims(centroid_vector) AS dims '
            f'FROM "{schema}".anchors ORDER BY anchor_id'
        )
        assert len(anchor_rows) == 3
        assert all(row["total_count"] == 1 for row in anchor_rows)
        assert all(row["dims"] == 1024 for row in anchor_rows)

        # event_atoms.anchor_id and events.data were preserved verbatim.
        preserved = await conn.fetch(
            f'SELECT occurrence_id, anchor_id FROM "{schema}".event_atoms ORDER BY occurrence_id'
        )
        assert [row["anchor_id"] for row in preserved] == [anchor_ids["张三"], anchor_ids["修复"], anchor_ids["服务器"]]

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


@requires_remote
@remote_group
async def test_remote_gate_refuses_mismatched_and_rebuilding_generations():
    """The shared gate refuses a mismatched/rebuilding generation before any bge call."""
    import asyncpg

    from hindsight_api.engine.retain.noesis_ingest import IdentitySpec, _embedding_space_profile_gate

    tunnel = _SshTunnel()
    conn = None
    try:
        conn = await asyncpg.connect(host="127.0.0.1", port=tunnel.port, user="postgres", database="noesis")
        outer_tx = conn.transaction()
        await outer_tx.start()
        schema = f"noesis_gate_test_{uuid.uuid4().hex[:12]}"
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(
            f'CREATE TABLE "{schema}".atoms ('
            "atom_id BIGINT PRIMARY KEY, text TEXT NOT NULL, atom_type CHAR(1) NOT NULL, embedding vector(1024))"
        )
        await conn.execute(
            f'CREATE TABLE "{schema}".anchors ('
            "anchor_id BIGINT PRIMARY KEY, atom_id BIGINT NOT NULL, centroid_vector vector(1024) NOT NULL, "
            "total_count BIGINT NOT NULL DEFAULT 0, status CHAR(1) NOT NULL DEFAULT 'A')"
        )
        await conn.execute(
            f'CREATE TABLE "{schema}".embedding_profiles ('
            "embedding_kind TEXT PRIMARY KEY, model_name TEXT NOT NULL, model_revision TEXT NOT NULL, "
            "dimension INTEGER NOT NULL, status TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        await conn.execute(
            f'INSERT INTO "{schema}".embedding_profiles '
            "(embedding_kind, model_name, model_revision, dimension, status) "
            "VALUES ('identity', 'bge-m3', 'bge-m3-1024-v1', 1024, 'ready')"
        )
        spec = IdentitySpec(
            base_url="http://10.0.0.8:8010", model="bge-m3", revision="bge-m3-1024-v2", dimension=1024
        )

        gate = await _embedding_space_profile_gate(_SingleConnPool(conn), schema, spec)
        assert gate.allowed is False
        assert gate.reason == "generation_mismatch"

        await conn.execute(f'UPDATE "{schema}".embedding_profiles SET status = \'rebuilding\'')
        gate = await _embedding_space_profile_gate(_SingleConnPool(conn), schema, spec)
        assert gate.allowed is False
        assert gate.reason == "rebuilding"

        await outer_tx.rollback()
    finally:
        if conn is not None:
            await conn.close()
        tunnel.close()
