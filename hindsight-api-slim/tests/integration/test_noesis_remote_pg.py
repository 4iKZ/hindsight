"""Remote PostgreSQL smoke test for Noesis event ingestion (requirement 02 §17.8).

Runs the real store layer (real asyncpg, real noesis_core constraints) against
the remote ``noesis`` database through an SSH tunnel. No LLM is involved: the
extraction seam serves golden components from requirement 01.

Everything happens inside ONE outer transaction on a single shared connection
so the final ROLLBACK leaves zero business rows behind (sequence increments
are accepted). Gated: skipped unless ``NOESIS_REMOTE_TEST=1`` and the SSH
credentials env vars are present — the normal regression run never touches
the remote database.

Credentials come exclusively from the environment; nothing is written to the
source, fixtures, or logs.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

pytest.importorskip("paramiko")
pytest.importorskip("asyncpg")

from hindsight_api.engine.retain import noesis_ingest  # noqa: E402
from tests.noesis_fakes import (  # noqa: E402
    FakeExtractOnceFactory,
    FakeIdentityClient,
    golden_fact_recursive,
    golden_fact_time,
    identity_factory_for,
    llm_config,
    noesis_config,
)

_EXPECTED_HOST_KEY = "SHA256:fYPgM4a2OY1ZRhdQbx2z2YjiQ9bOMx4zo/c1ewn+WCs"
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _remote_enabled() -> bool:
    return (
        os.environ.get("NOESIS_REMOTE_TEST") == "1"
        and bool(os.environ.get("NOESIS_SSH_HOST"))
        and bool(os.environ.get("NOESIS_SSH_PORT"))
        and bool(os.environ.get("NOESIS_SSH_USER"))
        and bool(os.environ.get("NOESIS_SSH_PW"))
    )


requires_remote = pytest.mark.skipif(not _remote_enabled(), reason="NOESIS_REMOTE_TEST/SSH env not set")


# ---------------------------------------------------------------------------
# SSH tunnel: local port -> remote localhost:5432 (host key pinned)
# ---------------------------------------------------------------------------

class _SshTunnel:
    """Minimal paramiko direct-tcpip forwarder (one local port, N sockets)."""

    def __init__(self) -> None:
        import base64
        import hashlib

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


# ---------------------------------------------------------------------------
# Single-connection pool: everything stays inside one outer transaction
# ---------------------------------------------------------------------------

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


async def _ingest_two_golden_facts(pool, analyzer):
    """Golden fact 1 (昨天/在超市 modifiers) + golden fact 3 (repeated 小明,
    nested predicate target chains) as two components of one content item."""

    def fake_extract(text, *, extract_once):
        from hyperextract.noesis import ExtractionOutcome

        return ExtractionOutcome(
            components=[golden_fact_time(), golden_fact_recursive()], alerts=[], attempts=1
        )

    original = noesis_ingest.extract_noesis_components
    noesis_ingest.extract_noesis_components = fake_extract
    # A fake identity client keeps this requirement-02 smoke deterministic and
    # offline: real bge + PG coverage lives in test_noesis_identity_remote.py.
    # Req-03 embeds every new E/P atom, so atoms.embedding is now non-NULL.
    try:
        await noesis_ingest.ingest_noesis_batch(
            [
                {
                    "content": "昨天妈妈在超市买了苹果。小明没写作业后揍了自己。",
                    "event_date": datetime(2026, 9, 5, 2, 0, 0, tzinfo=UTC),
                    "document_id": "smoke-doc",
                }
            ],
            "smoke-bank",
            noesis_config(),
            llm_config=llm_config(),
            analyzer=analyzer,
            extract_once_factory=FakeExtractOnceFactory(),
            pool_factory=_async_return(pool),
            identity_client_factory=identity_factory_for(FakeIdentityClient()),
        )
    finally:
        noesis_ingest.extract_noesis_components = original


def _async_return(value):
    async def factory(_config):
        return value

    return factory


class _YesterdayAnalyzer:
    """Offline analyzer: 昨天 resolves to the 2026-09-04 Shanghai day."""

    def analyze(self, query, reference_date=None):
        if query != "昨天":
            return None
        from hindsight_api.engine.query_analyzer import TemporalConstraint

        return TemporalConstraint(
            start_date=datetime(2026, 9, 4, 0, 0),
            end_date=datetime(2026, 9, 4, 23, 59, 59, 999999),
        )


# ---------------------------------------------------------------------------
# The smoke test
# ---------------------------------------------------------------------------

@requires_remote
async def test_remote_noesis_smoke():
    import asyncpg

    tunnel = _SshTunnel()
    conn = None
    try:
        conn = await asyncpg.connect(host="127.0.0.1", port=tunnel.port, user="postgres", database="noesis")

        # 0. Requirement 03: migration 003 is idempotent (CREATE TABLE IF NOT
        #    EXISTS). Ensured once here so the ingest preflight sees the profile
        #    gate table. (Its two-run idempotency is asserted separately in
        #    test_noesis_ddl_contract.py; a second multi-statement CREATE on the
        #    same connection trips pg_type_typname_nsp_index, so we skip when the
        #    table already exists.)
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='noesis_core' AND table_name='embedding_profiles')"
        )
        if not exists:
            migration_003 = _REPO_ROOT / "docs" / "db" / "migrations" / "003-noesis-identity-embedding-profile.sql"
            await conn.execute(migration_003.read_text(encoding="utf-8"))

        # 1. Extensions
        extensions = {
            row["extname"]: row["extversion"]
            for row in await conn.fetch(
                "SELECT extname, extversion FROM pg_extension "
                "WHERE extname = ANY($1)",
                ["vector", "roaringbitmap", "timescaledb", "pg_ripple"],
            )
        }
        assert set(extensions) == {"vector", "roaringbitmap", "timescaledb", "pg_ripple"}, extensions

        # 2. Migration constraints present
        constraints = {
            row["conname"]
            for row in await conn.fetch(
                "SELECT conname FROM pg_constraint WHERE conname = ANY($1)",
                ["atoms_text_type_unique", "events_ingestion_key_time_unique", "ingestion_alerts_dedupe_key_unique"],
            )
        }
        assert constraints == {
            "atoms_text_type_unique",
            "events_ingestion_key_time_unique",
            "ingestion_alerts_dedupe_key_unique",
        }

        # 3. Outer transaction: everything below rolls back at the end.
        # (asyncpg forbids manual BEGIN via execute(); its transaction object
        # nests the ingest transactions as SAVEPOINTs automatically.)
        outer_tx = conn.transaction()
        await outer_tx.start()

        pool = _SingleConnPool(conn)

        # 4. Write the two golden facts (modifiers, repeated atom, target chains)
        await _ingest_two_golden_facts(pool, _YesterdayAnalyzer())

        # 5. Verify events / atoms / event_atoms / data
        events = await conn.fetch(
            "SELECT event_id, event_time, ingestion_key, data FROM noesis_core.events ORDER BY event_id"
        )
        assert len(events) == 2
        by_predicate = {json.loads(e["data"])["component"]["tree"]["predicate"]: e for e in events}
        assert set(by_predicate) == {"买", "揍"}

        buy_event = by_predicate["买"]
        beat_event = by_predicate["揍"]
        # 昨天 resolved to the Shanghai day start → 2026-09-03T16:00:00Z
        assert buy_event["event_time"] == datetime(2026, 9, 3, 16, 0, 0, tzinfo=UTC)
        beat_data = json.loads(beat_event["data"])
        # no temporal modifier in the second fact → observed_at
        assert beat_data["time_resolution"]["strategy"] == "observed_at"
        assert beat_event["event_time"] == datetime(2026, 9, 5, 2, 0, 0, tzinfo=UTC)
        assert beat_data["contract_version"] == "noesis-event-closure-v1"
        assert beat_data["bank_id"] == "smoke-bank"
        assert beat_data["source_text"] == "昨天妈妈在超市买了苹果。小明没写作业后揍了自己。"

        atoms = {
            row["text"] + row["atom_type"]: row
            for row in await conn.fetch("SELECT atom_id, text, atom_type, support_count, embedding FROM noesis_core.atoms")
        }
        # 买-fact: 昨天 E, 妈妈 E, 在超市 E, 买 P, 苹果 E
        # 揍-fact: 小明 E, 没写 P, 作业 E, 揍 P
        assert set(atoms) == {"昨天E", "妈妈E", "在超市E", "买P", "苹果E", "小明E", "没写P", "作业E", "揍P"}
        assert all(row["support_count"] == 1 for row in atoms.values())  # repeated 小明 → one support
        # Requirement 03: every new E/P atom now embeds BGE(pure literal) on
        # creation (the fake identity client yields a deterministic 1024-dim
        # vector), so embedding is non-NULL for E/P. Real-bge coverage is in
        # test_noesis_identity_remote.py; G atoms stay NULL by contract.
        assert all(row["embedding"] is not None for row in atoms.values()), "E/P atoms must carry a vector"

        beat_atoms = await conn.fetch(
            "SELECT occurrence_id, atom_id, anchor_id, role_type, head_occurrence_id "
            "FROM noesis_core.event_atoms WHERE event_id = $1 ORDER BY occurrence_id",
            beat_event["event_id"],
        )
        assert len(beat_atoms) == 5  # 小明 twice → two occurrences
        xiaoming_atom = atoms["小明E"]["atom_id"]
        assert beat_atoms[0]["atom_id"] == xiaoming_atom and beat_atoms[4]["atom_id"] == xiaoming_atom
        assert all(row["anchor_id"] is None for row in beat_atoms)
        heads = {row["occurrence_id"]: row["head_occurrence_id"] for row in beat_atoms}
        assert heads == {1: 4, 2: 4, 3: 2, 4: None, 5: 4}  # mechanical target_occ mapping

        # 6. Replay: same input twice → still one event per fact, support +0
        await _ingest_two_golden_facts(pool, _YesterdayAnalyzer())
        replay_events = await conn.fetch("SELECT count(*) c FROM noesis_core.events")
        assert replay_events[0]["c"] == 2
        replay_support = await conn.fetchval(
            "SELECT support_count FROM noesis_core.atoms WHERE text='小明' AND atom_type='E'"
        )
        assert replay_support == 1
        replay_atoms = await conn.fetch("SELECT count(*) c FROM noesis_core.event_atoms")
        assert replay_atoms[0]["c"] == 10  # 5 per fact, unchanged

        # 7. Failure injection: break the event_atoms insert for a FRESH fact
        #    (a replay would skip the write path entirely) → full rollback,
        #    event_ingest_failed alert written, batch continues.
        original_sql = noesis_ingest._EVENT_ATOM_INSERT
        original_extract = noesis_ingest.extract_noesis_components
        noesis_ingest._EVENT_ATOM_INSERT = "INSERT INTO {s}.event_atoms (event_id, occurrence_id, nonexistent_column) VALUES ($1, $2, $3)"
        try:
            def fresh_extract(text, *, extract_once):
                from hyperextract.noesis import ExtractionOutcome

                return ExtractionOutcome(components=[golden_fact_recursive()], alerts=[], attempts=1)

            noesis_ingest.extract_noesis_components = fresh_extract
            await noesis_ingest.ingest_noesis_batch(
                [{"content": "全新内容触发全新写入路径。", "event_date": datetime(2026, 9, 6, 2, 0, 0, tzinfo=UTC)}],
                "smoke-bank",
                noesis_config(),
                llm_config=llm_config(),
                analyzer=_YesterdayAnalyzer(),
                extract_once_factory=FakeExtractOnceFactory(),
                pool_factory=_async_return(pool),
                identity_client_factory=identity_factory_for(FakeIdentityClient()),
            )
        finally:
            noesis_ingest._EVENT_ATOM_INSERT = original_sql
            noesis_ingest.extract_noesis_components = original_extract

        failed_alerts = await conn.fetch(
            "SELECT alert_code, severity, stage FROM noesis_core.ingestion_alerts WHERE alert_code = 'event_ingest_failed'"
        )
        assert len(failed_alerts) == 1  # the fresh fact failed and rolled back
        after_failure_events = await conn.fetch("SELECT count(*) c FROM noesis_core.events")
        assert after_failure_events[0]["c"] == 2  # no new events, old intact

        # 8. Rollback the outer transaction: zero business rows remain.
        await outer_tx.rollback()
        for table in ("events", "event_atoms", "ingestion_alerts"):
            remaining = await conn.fetchval(f"SELECT count(*) FROM noesis_core.{table}")
            assert remaining == 0, f"{table} rows leaked: {remaining}"
        remaining_atoms = await conn.fetchval("SELECT count(*) FROM noesis_core.atoms")
        assert remaining_atoms == 0, f"atoms rows leaked: {remaining_atoms}"
    finally:
        if conn is not None:
            await conn.close()
        tunnel.close()
