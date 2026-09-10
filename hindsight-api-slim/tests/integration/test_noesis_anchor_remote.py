"""Remote bge-m3 + PostgreSQL anchor-routing smoke (requirement 05 §13.7/§13.4).

Group A runs the REAL write path — real ``NoesisEmbeddingClient`` against the
live bge-m3 service, real asyncpg, real ``noesis_core`` with migration 005
applied — through the fake-LLM extraction seam serving the four frozen §13.7
fact components (妈妈让小明打酱油 / 苹果公司销售苹果 / 张三和李四共同修复服务器
与数据库 / 小明吃苹果苹果发布手机), plus one duplicate of the first component in
the same batch to prove reuse/create/EMA. Everything happens inside ONE outer
transaction on a single shared connection, so the final ROLLBACK leaves zero
business rows (sequence increments accepted).

Group B proves the requirement 05 §6.6/§13.4 concurrency contract on real
PostgreSQL row locks: two INDEPENDENT connections from a real asyncpg pool
call ``route_anchors`` concurrently on unique ``noesis05-test-`` literals —
no duplicate first buckets, no lost EMA updates, parallel different atoms —
with explicit cleanup (atoms DELETE cascades anchors via FK).

Gated: skipped unless ``NOESIS_REMOTE_TEST=1`` and the SSH env vars are
present. Credentials come exclusively from the environment; nothing is written
to the source, fixtures, or logs.
"""

from __future__ import annotations

import asyncio
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

from hyperextract.noesis import ExtractionOutcome, FactComponent  # noqa: E402

from hindsight_api.engine.retain import noesis_ingest  # noqa: E402
from hindsight_api.engine.retain.noesis_anchor import (  # noqa: E402
    AnchorPlan,
    OccurrenceRoute,
    PredicateFrame,
    parse_centroid_text,
    route_anchors,
)
from hindsight_api.engine.retain.noesis_embedding import NoesisEmbeddingClient  # noqa: E402
from tests.noesis_fakes import (  # noqa: E402
    FakeExtractOnceFactory,
    llm_config,
    noesis_config,
)

_EXPECTED_HOST_KEY = "SHA256:fYPgM4a2OY1ZRhdQbx2z2YjiQ9bOMx4zo/c1ewn+WCs"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_MIGRATION_003 = _REPO_ROOT / "docs" / "db" / "migrations" / "003-noesis-identity-embedding-profile.sql"
_REAL_BGE_URL = "http://10.0.0.8:8010"
_SCHEMA = "noesis_core"
_TABLES = ("events", "atoms", "anchors", "event_atoms")


def _remote_enabled() -> bool:
    return (
        os.environ.get("NOESIS_REMOTE_TEST") == "1"
        and all(os.environ.get(k) for k in ("NOESIS_SSH_HOST", "NOESIS_SSH_PORT", "NOESIS_SSH_USER", "NOESIS_SSH_PW"))
    )


requires_remote = pytest.mark.skipif(not _remote_enabled(), reason="NOESIS_REMOTE_TEST/SSH env not set")

# Group A asserts table counts against a pre-batch snapshot and Group B holds
# committed noesis05-test- rows between its INSERT and its explicit cleanup, so
# every remote noesis_core test must serialize under pytest-xdist (otherwise a
# parallel sibling's visible window breaks the zero-residue assertions).
remote_group = pytest.mark.xdist_group("noesis_remote")


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
    """Pool facade whose acquire always returns the same connection, so every
    ingest transaction nests (savepoints) inside the test's outer transaction."""

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


async def _real_bge_client() -> NoesisEmbeddingClient:
    return NoesisEmbeddingClient(
        base_url=_REAL_BGE_URL,
        model="bge-m3",
        revision="bge-m3-1024-v1",
        dimension=1024,
        timeout_seconds=8.0,
        max_retries=1,
    )


def _cosine_distance(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 1.0 - dot / (na * nb) if na and nb else 1.0


def _assert_close(actual: list[float], expected: list[float], *, what: str, tolerance: float = 2e-6) -> None:
    """Per-component closeness: pgvector stores float4 (~1e-8 per component
    here), the frozen EMA is computed in float64, so a lost-update-sized
    divergence (0.09*|c0-c1| ~ 1e-3) is far above the tolerance."""
    assert len(actual) == len(expected), f"{what}: dimension {len(actual)} != {len(expected)}"
    worst = max(abs(x - e) for x, e in zip(actual, expected))
    assert worst <= tolerance, f"{what}: worst component deviation {worst} > {tolerance}"


class _NullAnalyzer:
    """The frozen §13.7 components carry no temporal modifier; keep the time
    resolution on observed_at without a real dateparser dependency."""

    def analyze(self, query, reference_date=None):
        return None


# ---------------------------------------------------------------------------
# Group A golden facts: the four frozen §13.7 components (+ one duplicate)
# ---------------------------------------------------------------------------

def _atom(pos, text, type_, role, target):
    return {"pos": pos, "text": text, "type": type_, "role": role, "target_occ": target, "resolved": None}


_CASE_MAMA = [  # 妈妈让小明打酱油: Frame(2)="妈妈 让 小明", Frame(4)="打 酱油"
    _atom(1, "妈妈", "E", "agent", 2),
    _atom(2, "让", "P", "predicate", None),
    _atom(3, "小明", "E", "patient", 2),
    _atom(4, "打", "P", "predicate", 2),
    _atom(5, "酱油", "E", "patient", 4),
]
_CASE_SALE = [  # 苹果公司销售苹果: Frame(2)="苹果公司 销售 苹果"
    _atom(1, "苹果公司", "E", "agent", 2),
    _atom(2, "销售", "P", "predicate", None),
    _atom(3, "苹果", "E", "patient", 2),
]
_CASE_FIX = [  # 张三和李四共同修复服务器与数据库: Frame(3)="张三 李四 修复 服务器 数据库"
    _atom(1, "张三", "E", "agent", 3),
    _atom(2, "李四", "E", "agent", 3),
    _atom(3, "修复", "P", "predicate", None),
    _atom(4, "服务器", "E", "patient", 3),
    _atom(5, "数据库", "E", "patient", 3),
]
_CASE_MULTIFRAME = [  # 小明吃苹果，苹果发布手机: Frame(2)="小明 吃 苹果", Frame(5)="苹果 发布 手机"
    _atom(1, "苹果", "E", "patient", 2),
    _atom(2, "吃", "P", "predicate", None),
    _atom(3, "小明", "E", "agent", 2),
    _atom(4, "苹果", "E", "agent", 5),
    _atom(5, "发布", "P", "predicate", 2),
    _atom(6, "手机", "E", "patient", 5),
]


def _fact(atoms) -> FactComponent:
    root = next(a for a in atoms if a["role"] == "predicate" and a["target_occ"] is None)
    direct = [a for a in atoms if a["target_occ"] == root["pos"]]
    return FactComponent.model_validate(
        {
            "utterance_type": "fact",
            "atoms": atoms,
            "tree": {
                "predicate": root["text"],
                "agent": [{"text": a["text"], "modifier": [], "implied": False} for a in direct if a["role"] == "agent"],
                "patient": [{"text": a["text"], "modifier": [], "implied": False} for a in direct if a["role"] == "patient"],
                "modifier": [a["text"] for a in direct if a["role"] == "modifier"],
                "nested": [],
                "conditional": [],
            },
        }
    )


async def _table_count(conn, table: str) -> int:
    return await conn.fetchval(f"SELECT count(*) FROM noesis_core.{table}")


@requires_remote
@remote_group
async def test_remote_anchor_full_pipeline_smoke_and_rollback():
    """Group A: real bge context+identity vectors, real PostgreSQL, the four
    frozen §13.7 components plus a duplicated first component (reuse/EMA), all
    inside one outer transaction that ROLLs BACK leaving zero residue."""
    import asyncpg

    tunnel = _SshTunnel()
    conn = None
    client = None
    try:
        conn = await asyncpg.connect(host="127.0.0.1", port=tunnel.port, user="postgres", database="noesis")

        # Same migration-003 gate bootstrap as the other remote tests (only
        # when the profile table is missing; multi-statement CREATE twice on
        # one connection trips pg_type_typname_nsp_index).
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='noesis_core' AND table_name='embedding_profiles')"
        )
        if not exists:
            await conn.execute(_MIGRATION_003.read_text(encoding="utf-8"))

        snapshot = {table: await _table_count(conn, table) for table in _TABLES}

        outer_tx = conn.transaction()
        await outer_tx.start()
        pool = _SingleConnPool(conn)

        # The frozen batch: components 1-4 plus a duplicate of component 1 in
        # the SAME batch — its identical context vectors must reuse (not
        # re-create) every anchor of the first pass with an EMA update.
        components = [
            _fact(_CASE_MAMA),
            _fact(_CASE_SALE),
            _fact(_CASE_FIX),
            _fact(_CASE_MULTIFRAME),
            _fact(_CASE_MAMA),
        ]

        def fake_extract(text, *, extract_once):
            return ExtractionOutcome(components=list(components), alerts=[], attempts=1)

        original_extract = noesis_ingest.extract_noesis_components
        noesis_ingest.extract_noesis_components = fake_extract
        client = await _real_bge_client()
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
                        "content": "妈妈让小明打酱油。苹果公司销售苹果。张三和李四共同修复服务器与数据库。小明吃苹果，苹果发布手机。",
                        "event_date": datetime(2026, 9, 5, 2, 0, 0, tzinfo=UTC),
                        "document_id": "anchor-smoke-doc",
                    }
                ],
                "anchor-smoke-bank",
                cfg,
                llm_config=llm_config(),
                analyzer=_NullAnalyzer(),
                extract_once_factory=FakeExtractOnceFactory(),
                pool_factory=_async_return(pool),
                embedding_client_factory=_async_return(client),
            )
        finally:
            noesis_ingest.extract_noesis_components = original_extract

        # 1. Six events? No — five components -> five events, all with a NULL
        #    events.context_embedding (§8.4: the frame vector never lands here).
        events = await conn.fetch(
            "SELECT event_id, (data->>'component_index')::int AS ci, context_embedding FROM noesis_core.events ORDER BY event_id"
        )
        assert [row["ci"] for row in events] == [0, 1, 2, 3, 4]
        assert all(row["context_embedding"] is None for row in events)

        # 2. Every event_atom row (24 = 5+3+5+6+5 occurrences) carries a
        #    concrete anchor that exists in noesis_core.anchors; E and P alike.
        rows = await conn.fetch(
            "SELECT ea.event_id, ea.occurrence_id, ea.atom_id, ea.anchor_id, a.atom_type "
            "FROM noesis_core.event_atoms ea JOIN noesis_core.atoms a USING (atom_id) "
            "ORDER BY ea.event_id, ea.occurrence_id"
        )
        assert len(rows) == 24
        assert all(row["anchor_id"] is not None for row in rows), "a successful occurrence carried a NULL anchor"
        orphans = await conn.fetchval(
            "SELECT count(*) FROM noesis_core.event_atoms ea "
            "LEFT JOIN noesis_core.anchors an ON an.anchor_id = ea.anchor_id WHERE an.anchor_id IS NULL"
        )
        assert orphans == 0
        atom_types = {row["atom_type"] for row in rows}
        assert atom_types <= {b"E", b"P"}  # asyncpg decodes the one-byte "char" enum as bytes
        atoms_table = await conn.fetch("SELECT atom_id, atom_type FROM noesis_core.atoms")
        assert {row["atom_type"] for row in atoms_table} == {b"E", b"P"}  # all 16 literals are routable E/P
        anchored = {row["atom_id"] for row in await conn.fetch("SELECT atom_id FROM noesis_core.anchors")}
        assert anchored >= {row["atom_id"] for row in atoms_table}, "an E/P atom has no anchor at all"

        # 3. Frame routing per component: the two 妈妈 events (component_index
        #    0 and 4) each land 5 anchored occurrences across both frames.
        mama_event_ids = [row["event_id"] for row in events if row["ci"] in (0, 4)]
        assert len(mama_event_ids) == 2
        for event_id in mama_event_ids:
            occurrences = await conn.fetch(
                "SELECT occurrence_id, atom_id, anchor_id FROM noesis_core.event_atoms "
                "WHERE event_id = $1 ORDER BY occurrence_id",
                event_id,
            )
            assert len(occurrences) == 5
            assert all(row["anchor_id"] is not None for row in occurrences)
        # Multi agent/patient: the 修复 event carries 5 anchored occurrences.
        fix_event_id = events[2]["event_id"]
        fix_rows = await conn.fetch(
            "SELECT anchor_id FROM noesis_core.event_atoms WHERE event_id = $1 ORDER BY occurrence_id", fix_event_id
        )
        assert len(fix_rows) == 5  # 张三 李四 修复 服务器 数据库

        # 4. Anchor table state: every (atom, frame) route either creates a
        #    bucket (count=1) or reuses one (count+=1), so the total sampled
        #    count equals the number of routed (atom, frame) pairs: 24.
        total_count_sum = await conn.fetchval("SELECT sum(total_count) FROM noesis_core.anchors")
        assert total_count_sum == 24
        anchor_rows_count = await conn.fetchval("SELECT count(*) FROM noesis_core.anchors")
        assert anchor_rows_count >= len(atoms_table)  # one bucket per atom minimum

        # 5. reuse/create/EMA on literals owned by exactly one component.
        #    妈妈/让 ride Frame(2) twice (count=2, centroid unchanged);
        #    打/酱油 ride Frame(4) twice; 销售 group rides Frame(2) once;
        #    修复 group rides its single frame once with the SAME context.
        vec_mama = await client.embed_context("妈妈 让 小明")
        vec_da = await client.embed_context("打 酱油")
        vec_sale = await client.embed_context("苹果公司 销售 苹果")
        vec_fix = await client.embed_context("张三 李四 修复 服务器 数据库")

        async def anchors_of(text: str, atom_type: str):
            return await conn.fetch(
                "SELECT an.total_count, an.centroid_vector::text AS centroid_text "
                "FROM noesis_core.anchors an JOIN noesis_core.atoms a USING (atom_id) "
                "WHERE a.text = $1 AND a.atom_type = $2::text::\"char\"",
                text,
                atom_type,
            )

        for text, atom_type, vector, count in (
            ("妈妈", "E", vec_mama, 2),
            ("让", "P", vec_mama, 2),
            ("酱油", "E", vec_da, 2),
            ("打", "P", vec_da, 2),
            ("苹果公司", "E", vec_sale, 1),
            ("销售", "P", vec_sale, 1),
            ("张三", "E", vec_fix, 1),
            ("李四", "E", vec_fix, 1),
            ("修复", "P", vec_fix, 1),
            ("服务器", "E", vec_fix, 1),
            ("数据库", "E", vec_fix, 1),
            ("吃", "P", await client.embed_context("小明 吃 苹果"), 1),
            ("发布", "P", await client.embed_context("苹果 发布 手机"), 1),
            ("手机", "E", await client.embed_context("苹果 发布 手机"), 1),
        ):
            found = await anchors_of(text, atom_type)
            assert len(found) == 1, f"{text}/{atom_type}: expected exactly one bucket, got {len(found)}"
            assert found[0]["total_count"] == count, f"{text}/{atom_type}: total_count={found[0]['total_count']}, want {count}"
            _assert_close(parse_centroid_text(found[0]["centroid_text"]), vector, what=f"{text}/{atom_type} centroid")

        # 6. Same-form different-frame: 苹果 appears in three routes (sale
        #    Frame(2), multi-frame Frame(2) and Frame(5)); 小明 in three (mama
        #    Frame(2) twice, multi-frame Frame(2) once). Whatever bge decides
        #    about reuse across those contexts, every occurrence stayed
        #    anchored (assertion 2) and the totals balance (assertion 4).
        for text in ("苹果", "小明"):
            apple_counts = await conn.fetchval(
                "SELECT sum(an.total_count) FROM noesis_core.anchors an "
                "JOIN noesis_core.atoms a USING (atom_id) WHERE a.text = $1",
                text,
            )
            assert apple_counts == 3, f"{text}: routed three times, sampled total {apple_counts} != 3"

        # 7. ROLLBACK the outer transaction: zero business rows remain.
        await outer_tx.rollback()
        for table in _TABLES:
            remaining = await _table_count(conn, table)
            assert remaining == snapshot[table], f"{table} leaked: {remaining} != {snapshot[table]}"
    finally:
        if client is not None:
            await client.aclose()
        if conn is not None:
            await conn.close()
        tunnel.close()


# ---------------------------------------------------------------------------
# Group B: real-PostgreSQL concurrency on route_anchors (§13.4)
# ---------------------------------------------------------------------------

def _single_occurrence_plan(literal: tuple[str, str], context_text: str) -> AnchorPlan:
    return AnchorPlan(
        frames={1: PredicateFrame(1, context_text)},
        occurrences=(OccurrenceRoute(1, literal, 1),),
        context_texts=(context_text,),
    )


async def _route_in_pool(pool, *, atom_ids, plan, context_vectors):
    async with pool.acquire() as conn:
        async with conn.transaction():
            return await route_anchors(conn, _SCHEMA, atom_ids=atom_ids, plan=plan, context_vectors=context_vectors)


async def _anchors_of(pool, atom_id: int):
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT anchor_id, total_count, centroid_vector::text AS centroid_text "
            "FROM noesis_core.anchors WHERE atom_id = $1 ORDER BY anchor_id",
            atom_id,
        )


@requires_remote
@remote_group
async def test_remote_anchor_concurrency_real_locks():
    """Group B (requirement 05 §13.4): C1 concurrent first-create yields one
    bucket; C2 concurrent reuses never lose an EMA update; C3 different atoms
    route in parallel. Real bge context vectors, two independent connections,
    explicit noesis05-test- cleanup."""
    import asyncpg

    tunnel = _SshTunnel()
    pool = None
    client = None
    snapshot: dict[str, int] = {}
    try:
        pool = await asyncpg.create_pool(
            host="127.0.0.1", port=tunnel.port, user="postgres", database="noesis", min_size=2, max_size=4
        )
        client = await _real_bge_client()
        async with pool.acquire() as conn:
            snapshot = {table: await _table_count(conn, table) for table in _TABLES}

        run = uuid.uuid4().hex[:8]
        literals = {
            name: (f"noesis05-test-{run}-{name}", "E") for name in ("alpha", "beta", "gamma1", "gamma2")
        }
        ctx_alpha = f"noesis05-test-{run}-alpha 甲 修复"
        ctx_beta0 = f"noesis05-test-{run}-beta 甲 修复"
        ctx_beta1 = f"noesis05-test-{run}-beta 乙 修复"
        ctx_gamma1 = f"noesis05-test-{run}-gamma1 甲 修复"
        ctx_gamma2 = f"noesis05-test-{run}-gamma2 乙 修复"
        v_alpha = await client.embed_context(ctx_alpha)
        v_beta0 = await client.embed_context(ctx_beta0)
        v_beta1 = await client.embed_context(ctx_beta1)
        v_gamma1 = await client.embed_context(ctx_gamma1)
        v_gamma2 = await client.embed_context(ctx_gamma2)

        async with pool.acquire() as conn:
            async with conn.transaction():
                atom_ids = {}
                for literal in literals.values():
                    row = await conn.fetchrow(
                        "INSERT INTO noesis_core.atoms (text, atom_type) VALUES ($1, $2::text::\"char\") RETURNING atom_id",
                        literal[0],
                        literal[1],
                    )
                    atom_ids[literal] = row["atom_id"]

        # -- C1: two concurrent first routes on the same atom + context ------
        plan_alpha = _single_occurrence_plan(literals["alpha"], ctx_alpha)
        alpha_routes = await asyncio.gather(
            _route_in_pool(pool, atom_ids=atom_ids, plan=plan_alpha, context_vectors={ctx_alpha: v_alpha}),
            _route_in_pool(pool, atom_ids=atom_ids, plan=plan_alpha, context_vectors={ctx_alpha: v_alpha}),
        )
        assert alpha_routes[0][(atom_ids[literals["alpha"]], 1)] == alpha_routes[1][(atom_ids[literals["alpha"]], 1)]
        c1 = await _anchors_of(pool, atom_ids[literals["alpha"]])
        assert len(c1) == 1, f"C1: duplicate first buckets: {len(c1)}"
        assert c1[0]["total_count"] == 2, f"C1: total_count={c1[0]['total_count']}, want 2"
        _assert_close(parse_centroid_text(c1[0]["centroid_text"]), v_alpha, what="C1 EMA centroid")

        # -- C2: concurrent hits on an existing bucket lose no EMA update -----
        plan_beta0 = _single_occurrence_plan(literals["beta"], ctx_beta0)
        await _route_in_pool(pool, atom_ids=atom_ids, plan=plan_beta0, context_vectors={ctx_beta0: v_beta0})
        seeded = await _anchors_of(pool, atom_ids[literals["beta"]])
        assert len(seeded) == 1 and seeded[0]["total_count"] == 1
        # The concurrent context must be within the reuse gate, otherwise the
        # gather below would prove parallel creates, not serialized EMA.
        assert _cosine_distance(v_beta0, v_beta1) <= 0.25, "bge contexts too far apart for the reuse gate"
        plan_beta1 = _single_occurrence_plan(literals["beta"], ctx_beta1)
        await asyncio.gather(
            _route_in_pool(pool, atom_ids=atom_ids, plan=plan_beta1, context_vectors={ctx_beta1: v_beta1}),
            _route_in_pool(pool, atom_ids=atom_ids, plan=plan_beta1, context_vectors={ctx_beta1: v_beta1}),
        )
        c2 = await _anchors_of(pool, atom_ids[literals["beta"]])
        assert len(c2) == 1, f"C2: bucket count {len(c2)} != 1"
        assert c2[0]["total_count"] == 3, f"C2: total_count={c2[0]['total_count']}, want 3"
        # Serialized EMAs: EMA(EMA(c0, c1), c1) = 0.81*c0 + 0.19*c1. A lost
        # update leaves EMA(c0, c1) = 0.9*c0 + 0.1*c1 — the per-component gap
        # 0.09*|c0-c1| is orders of magnitude above the float4 tolerance.
        expected_c2 = [0.81 * a + 0.19 * b for a, b in zip(v_beta0, v_beta1)]
        _assert_close(parse_centroid_text(c2[0]["centroid_text"]), expected_c2, what="C2 serialized EMA centroid")

        # -- C3: two different atoms route concurrently ----------------------
        plan_g1 = _single_occurrence_plan(literals["gamma1"], ctx_gamma1)
        plan_g2 = _single_occurrence_plan(literals["gamma2"], ctx_gamma2)
        await asyncio.gather(
            _route_in_pool(pool, atom_ids=atom_ids, plan=plan_g1, context_vectors={ctx_gamma1: v_gamma1}),
            _route_in_pool(pool, atom_ids=atom_ids, plan=plan_g2, context_vectors={ctx_gamma2: v_gamma2}),
        )
        g1 = await _anchors_of(pool, atom_ids[literals["gamma1"]])
        g2 = await _anchors_of(pool, atom_ids[literals["gamma2"]])
        assert len(g1) == 1 and g1[0]["total_count"] == 1
        assert len(g2) == 1 and g2[0]["total_count"] == 1
        assert g1[0]["anchor_id"] != g2[0]["anchor_id"], "different atoms must not share a bucket"
    finally:
        if pool is not None:
            try:
                async with pool.acquire() as conn:
                    # anchors follow via the atoms FK ON DELETE CASCADE
                    await conn.execute("DELETE FROM noesis_core.atoms WHERE text LIKE 'noesis05-test-%'")
                    if snapshot:
                        for table in _TABLES:
                            remaining = await _table_count(conn, table)
                            assert remaining == snapshot[table], (
                                f"{table} leaked: {remaining} != {snapshot[table]}"
                            )
            finally:
                await pool.close()
        if client is not None:
            await client.aclose()
        tunnel.close()
