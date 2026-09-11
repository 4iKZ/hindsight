"""Remote cooccurrence retention smoke (requirement 07 §11.8).

Every write happens in a throwaway TEMP schema created from the canonical DDL
(remapped), so the real ``noesis_core`` tables are never modified. The real
production function runs against the live pg_roaringbitmap 1.2 extension and
proves: touched buckets lose exactly the expired bits, untouched buckets keep
theirs, an emptied bucket keeps its row with an empty bitmap, repeated batches
are idempotent, the Neighbor sentinel is bit-for-bit unchanged, and a caller
rollback restores every pruned bucket. A second group proves the real row-lock
serialization of ``rb64_or`` versus ``rb64_andnot`` on one bucket. The real
``noesis_core`` schema is only ever probed read-only: the requirement 07
retention preflight (payload type, strict ``(atom_id, anchor_id)`` primary key,
callable ``rb64_*`` functions) must pass on the deployed database before any
smoke write. Finally the temp schema is dropped and the six ``noesis_core``
tables are proven count-identical to the pre-smoke snapshot.

Gated: skipped unless ``NOESIS_REMOTE_TEST=1`` plus the SSH env vars are
present. Credentials come exclusively from the environment; nothing is written
to the source, fixtures, or logs.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import random
import socket
import string
import threading
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

pytest.importorskip("paramiko")
asyncpg = pytest.importorskip("asyncpg")

from hindsight_api.engine.retain.noesis_bitmap import (  # noqa: E402
    BitmapWritePlan,
    CooccurrenceBitmapWrite,
    apply_bitmap_writes,
)
from hindsight_api.engine.retain.noesis_cooccurrence_retention import (  # noqa: E402
    CooccurrencePruneResult,
    check_cooccurrence_retention_ready,
    prune_cooccurrence_event_ids,
)

_EXPECTED_HOST_KEY = "SHA256:fYPgM4a2OY1ZRhdQbx2z2YjiQ9bOMx4zo/c1ewn+WCs"
_REPO_ROOT = Path(__file__).resolve().parents[4]  # wxs-noesis/
_BASE_SQL_PATH = _REPO_ROOT / "docs" / "db" / "noesis-stage1-schema.sql"
_SCHEMA = "noesis_core"
_TABLES = ("events", "atoms", "anchors", "event_atoms", "cooccurrence_bitmaps", "neighbor_bitmaps")


def _remote_enabled() -> bool:
    return (
        os.environ.get("NOESIS_REMOTE_TEST") == "1"
        and all(os.environ.get(k) for k in ("NOESIS_SSH_HOST", "NOESIS_SSH_PORT", "NOESIS_SSH_USER", "NOESIS_SSH_PW"))
    )


requires_remote = pytest.mark.skipif(not _remote_enabled(), reason="NOESIS_REMOTE_TEST/SSH env not set")

# Serialize with every other remote noesis test under pytest-xdist: the
# zero-residue count assertions must not race a parallel sibling's writes.
remote_group = pytest.mark.xdist_group("noesis_remote")


class _Remote:
    """SSH host-key-pinned helper that uploads a script and runs psql -f."""

    def __init__(self) -> None:
        import paramiko

        self.transport = paramiko.Transport((os.environ["NOESIS_SSH_HOST"], int(os.environ["NOESIS_SSH_PORT"])))
        self.transport.start_client(timeout=20)
        key = self.transport.get_remote_server_key()
        fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        if fingerprint != _EXPECTED_HOST_KEY:
            self.transport.close()
            raise RuntimeError("remote host key mismatch — refusing to connect")
        self.transport.auth_password(username=os.environ["NOESIS_SSH_USER"], password=os.environ["NOESIS_SSH_PW"])
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def _exec(self, command: str, timeout: int = 120) -> tuple[int, str, str]:
        channel = self.transport.open_session()
        channel.settimeout(timeout)
        channel.exec_command(command)
        out = channel.makefile().read().decode()
        err = channel.makefile_stderr().read().decode()
        rc = channel.recv_exit_status()
        channel.close()
        return rc, out, err

    def run_file(self, local_path: Path, *, remap: str | None = None) -> str:
        content = local_path.read_text(encoding="utf-8")
        if remap:
            content = content.replace("noesis_core", remap)
        remote = f"/tmp/nzr07_{uuid.uuid4().hex}.sql"
        with self.sftp.open(remote, "w") as handle:
            handle.write(content)
        rc, out, err = self._exec(
            f"/usr/local/pgsql18/bin/psql -U postgres -d noesis -v ON_ERROR_STOP=1 -f {remote}; "
            f"rc=$?; rm -f -- {remote}; "
            f"if [ $rc -eq 0 ]; then echo MIGRATION_OK; fi; exit $rc"
        )
        if rc != 0:
            raise RuntimeError(f"psql failed (rc={rc}): {err.strip()}")
        return "OK"

    def query(self, sql: str) -> list[str]:
        rc, out, err = self._exec(
            "/usr/local/pgsql18/bin/psql -U postgres -d noesis -At -c \"" + sql.replace('"', '\\"') + "\""
        )
        if rc != 0:
            raise RuntimeError(f"psql query failed (rc={rc}): {err.strip()}")
        return [line for line in out.splitlines() if line]

    def leftover_files(self) -> list[str]:
        _rc, out, _err = self._exec("ls /tmp/nzr07_* 2>/dev/null || true")
        return [line for line in out.splitlines() if line.strip()]

    def close(self) -> None:
        self.sftp.close()
        self.transport.close()


class _SshTunnel:
    """Minimal paramiko direct-tcpip forwarder (one local port, N sockets)."""

    def __init__(self) -> None:
        import paramiko

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

    def _forward(self, client) -> None:
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


async def _connect(tunnel: _SshTunnel):
    return await asyncpg.connect(host="127.0.0.1", port=tunnel.port, user="postgres", database="noesis")


def _temp_schema(prefix: str) -> str:
    return f"{prefix}_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"


async def _table_count(conn, table: str) -> int:
    return int(await conn.fetchval(f"SELECT count(*) FROM {_SCHEMA}.{table}"))


async def _insert_atom(conn, schema: str, text: str, atom_type: str) -> int:
    return int(
        await conn.fetchval(
            f"INSERT INTO {schema}.atoms (text, atom_type) VALUES ($1, $2::text::\"char\") RETURNING atom_id",
            text,
            atom_type,
        )
    )


async def _insert_anchor(conn, schema: str, atom_id: int) -> int:
    return int(
        await conn.fetchval(
            f"INSERT INTO {schema}.anchors (atom_id, centroid_vector) "
            f"VALUES ($1, ('[' || array_to_string(array_fill(0.1::real, ARRAY[1024]), ',') || ']')::vector) "
            f"RETURNING anchor_id",
            atom_id,
        )
    )


async def _insert_bucket(conn, schema: str, atom_id: int, anchor_id: int, bits: list[int]) -> None:
    await conn.execute(
        f"INSERT INTO {schema}.cooccurrence_bitmaps (atom_id, anchor_id, event_bitmap) "
        f"VALUES ($1, $2, rb64_build($3::bigint[]))",
        atom_id,
        anchor_id,
        bits,
    )


async def _insert_neighbor_sentinel(conn, schema: str, atom_id: int, anchor_id: int, bits: list[int]) -> None:
    await conn.execute(
        f"INSERT INTO {schema}.neighbor_bitmaps (atom_id, anchor_id, role_type, neighbor_bitmap) "
        f"VALUES ($1, $2, 'N'::text::\"char\", rb64_build($3::bigint[]))",
        atom_id,
        anchor_id,
        bits,
    )


async def _bucket_matches(conn, schema: str, atom_id: int, anchor_id: int, expected: list[int]) -> bool:
    """True when the stored set is exactly ``expected`` (cardinality + intersection)."""
    return bool(
        await conn.fetchval(
            f"SELECT rb64_cardinality(event_bitmap) = $3::bigint "
            f"AND rb64_and_cardinality(event_bitmap, rb64_build($4::bigint[])) = $3::bigint "
            f"FROM {schema}.cooccurrence_bitmaps WHERE atom_id = $1 AND anchor_id = $2",
            atom_id,
            anchor_id,
            len(expected),
            expected,
        )
    )


async def _neighbor_matches(conn, schema: str, atom_id: int, anchor_id: int, expected: list[int]) -> bool:
    return bool(
        await conn.fetchval(
            f"SELECT rb64_cardinality(neighbor_bitmap) = $3::bigint "
            f"AND rb64_and_cardinality(neighbor_bitmap, rb64_build($4::bigint[])) = $3::bigint "
            f"FROM {schema}.neighbor_bitmaps WHERE atom_id = $1 AND anchor_id = $2 AND role_type = 'N'::\"char\"",
            atom_id,
            anchor_id,
            len(expected),
            expected,
        )
    )


@requires_remote
@remote_group
async def test_remote_prune_smoke_cardinality_rollback_and_zero_residue():
    """§11.8: hit/miss/empty/idempotent/rollback/Neighbor sentinel/zero residue."""
    schema = _temp_schema("nzr7")
    remote = _Remote()
    tunnel = _SshTunnel()
    conn = None
    try:
        conn = await _connect(tunnel)
        production_snapshot = {table: await _table_count(conn, table) for table in _TABLES}

        # Read-only proof that the deployed production schema satisfies the
        # requirement 07 retention preflight (type, strict PK, callable rb64).
        await check_cooccurrence_retention_ready(conn, _SCHEMA)

        remote.query(f"CREATE SCHEMA {schema}")
        try:
            remote.run_file(_BASE_SQL_PATH, remap=schema)
            await check_cooccurrence_retention_ready(conn, schema)

            atom_a = await _insert_atom(conn, schema, "nzr07-a", "P")
            atom_b = await _insert_atom(conn, schema, "nzr07-b", "E")
            atom_c = await _insert_atom(conn, schema, "nzr07-c", "E")
            anchor_a = await _insert_anchor(conn, schema, atom_a)
            anchor_b = await _insert_anchor(conn, schema, atom_b)
            anchor_c = await _insert_anchor(conn, schema, atom_c)
            await _insert_bucket(conn, schema, atom_a, anchor_a, [1, 2, 3, 4])
            await _insert_bucket(conn, schema, atom_b, anchor_b, [2, 5])
            await _insert_bucket(conn, schema, atom_c, anchor_c, [6, 7])
            sentinel_bits = [900000001, 900000002]
            await _insert_neighbor_sentinel(conn, schema, atom_a, anchor_a, sentinel_bits)

            # A = {1,2,3,4}, B = {2,5}, C = {6,7}; expired = {2,3,99}.
            result = await prune_cooccurrence_event_ids(conn, schema, [2, 3, 99])
            assert result == CooccurrencePruneResult(requested_ids=3, unique_ids=3, updated_buckets=2), result
            assert await _bucket_matches(conn, schema, atom_a, anchor_a, [1, 4]), "hit bucket must lose exactly 2,3"
            assert await _bucket_matches(conn, schema, atom_b, anchor_b, [5]), "hit bucket must keep its survivor"
            assert await _bucket_matches(conn, schema, atom_c, anchor_c, [6, 7]), "untouched bucket must not change"

            # Fully emptied bucket keeps its row with an empty bitmap.
            emptied = await prune_cooccurrence_event_ids(conn, schema, [5])
            assert emptied.updated_buckets == 1
            assert await _bucket_matches(conn, schema, atom_b, anchor_b, []), "emptied bucket must stay as empty set"
            assert await conn.fetchval(
                f"SELECT count(*) FROM {schema}.cooccurrence_bitmaps WHERE atom_id = $1 AND anchor_id = $2",
                atom_b,
                anchor_b,
            ) == 1, "emptied bucket row must not be deleted"

            # Repeat of an already applied batch is a zero-update no-op.
            repeated = await prune_cooccurrence_event_ids(conn, schema, [2, 3, 5])
            assert repeated.updated_buckets == 0
            assert repeated.unique_ids == 3
            assert await _bucket_matches(conn, schema, atom_a, anchor_a, [1, 4])
            assert await _bucket_matches(conn, schema, atom_b, anchor_b, [])

            # Caller rollback restores every pruned bucket bit-for-bit.
            transaction = conn.transaction()
            await transaction.start()
            try:
                rolled = await prune_cooccurrence_event_ids(conn, schema, [1, 4])
                assert rolled.updated_buckets == 1
            finally:
                await transaction.rollback()
            assert await _bucket_matches(conn, schema, atom_a, anchor_a, [1, 4]), "rollback must restore pruned bits"

            # Neighbor sentinel is bit-for-bit unchanged through every path.
            assert await _neighbor_matches(conn, schema, atom_a, anchor_a, sentinel_bits), "Neighbor sentinel changed"
        finally:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")

        for table in _TABLES:
            assert await _table_count(conn, table) == production_snapshot[table], f"{table} leaked from the smoke"
        assert remote.leftover_files() == [], "remote temp SQL files leaked"
    finally:
        if conn is not None:
            await conn.close()
        tunnel.close()
        remote.close()


@requires_remote
@remote_group
async def test_remote_or_andnot_row_lock_interleaving():
    """§5.3/§11.6: a concurrent OR upsert and ANDNOT prune on one bucket
    serialize on the row lock; the new bit survives and only the expired bit
    disappears, regardless of lock order."""
    schema = _temp_schema("nzr7c")
    remote = _Remote()
    tunnel = _SshTunnel()
    conn = conn_a = conn_b = None
    try:
        conn = await _connect(tunnel)
        production_snapshot = {table: await _table_count(conn, table) for table in _TABLES}
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            remote.run_file(_BASE_SQL_PATH, remap=schema)
            atom_id = await _insert_atom(conn, schema, "nzr07-concurrent", "E")
            anchor_id = await _insert_anchor(conn, schema, atom_id)
            await _insert_bucket(conn, schema, atom_id, anchor_id, [1, 2])

            conn_a = await _connect(tunnel)
            conn_b = await _connect(tunnel)

            async def pruner():
                async with conn_a.transaction():
                    result = await prune_cooccurrence_event_ids(conn_a, schema, [1])
                    assert result.updated_buckets == 1
                    await asyncio.sleep(0.3)

            async def writer():
                async with conn_b.transaction():
                    await apply_bitmap_writes(
                        conn_b,
                        schema,
                        BitmapWritePlan(
                            cooccurrences=(CooccurrenceBitmapWrite(atom_id, anchor_id, 3),),
                            neighbors=(),
                        ),
                    )
                    await asyncio.sleep(0.3)

            await asyncio.gather(pruner(), writer())

            assert await _bucket_matches(conn, schema, atom_id, anchor_id, [2, 3]), (
                "OR/ANDNOT interleaving lost a bit"
            )
        finally:
            for connection in (conn_a, conn_b):
                if connection is not None:
                    await connection.close()
            conn_a = conn_b = None
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")

        for table in _TABLES:
            assert await _table_count(conn, table) == production_snapshot[table], f"{table} leaked from the smoke"
        assert remote.leftover_files() == [], "remote temp SQL files leaked"
    finally:
        for connection in (conn_a, conn_b, conn):
            if connection is not None:
                await connection.close()
        tunnel.close()
        remote.close()
