"""Remote bitmap write-maintenance smoke (requirement 06 §12.7).

Everything happens in a throwaway TEMP schema created from the canonical DDL
(remapped), so the real ``noesis_core`` tables are never written: the writer's
real ``rb64_build`` / ``rb64_or`` ON CONFLICT statements run against the live
roaringbitmap 1.2 extension, cardinality and role-row isolation are verified
with ``rb64_cardinality``, then the schema is dropped and the six
``noesis_core`` tables are proven byte-count-identical to the pre-smoke
snapshot. Group B proves the concurrent union contract on two independent
connections (no lost update, no deadlock).

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
    NeighborBitmapWrite,
    apply_bitmap_writes,
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

    def run_file(self, local_path: Path, *, remap: str | None = None) -> str:
        content = local_path.read_text(encoding="utf-8")
        if remap:
            content = content.replace("noesis_core", remap)
        remote = f"/tmp/r06_bitmap_{uuid.uuid4().hex}.sql"
        with self.sftp.open(remote, "w") as handle:
            handle.write(content)
        channel = self.transport.open_session()
        channel.settimeout(120)
        channel.exec_command(
            f"/usr/local/pgsql18/bin/psql -U postgres -d noesis -v ON_ERROR_STOP=1 -f {remote} "
            f"&& rm -f {remote} && echo MIGRATION_OK"
        )
        channel.makefile().read()
        err = channel.makefile_stderr().read().decode().strip()
        rc = channel.recv_exit_status()
        channel.close()
        if rc != 0:
            raise RuntimeError(f"psql failed (rc={rc}): {err}")
        return "OK"

    def query(self, sql: str) -> list[str]:
        channel = self.transport.open_session()
        channel.settimeout(60)
        channel.exec_command(
            "/usr/local/pgsql18/bin/psql -U postgres -d noesis -At -c \"" + sql.replace('"', '\\"') + "\""
        )
        out = channel.makefile().read().decode().strip()
        err = channel.makefile_stderr().read().decode().strip()
        rc = channel.recv_exit_status()
        channel.close()
        if rc != 0:
            raise RuntimeError(f"psql query failed (rc={rc}): {err}")
        return [line for line in out.splitlines() if line]

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


async def _table_count(conn, table: str) -> int:
    return int(await conn.fetchval(f"SELECT count(*) FROM {_SCHEMA}.{table}"))


def _temp_schema(prefix: str) -> str:
    return f"{prefix}_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"


def _plan(*, event_id: int, atom_id: int, anchor_id: int) -> BitmapWritePlan:
    # Sentinel neighbor bits are far outside the temp-schema atom sequence so
    # the real atoms inserted by the test can never alias them.
    return BitmapWritePlan(
        cooccurrences=(CooccurrenceBitmapWrite(atom_id, anchor_id, event_id),),
        neighbors=(
            NeighborBitmapWrite(atom_id, anchor_id, "N", (900000001, 900000002)),
            NeighborBitmapWrite(atom_id, anchor_id, "S", (900000001,)),
            NeighborBitmapWrite(atom_id, anchor_id, "O", (900000002,)),
        ),
    )


async def _insert_event(conn, schema: str) -> int:
    return int(
        await conn.fetchval(
            f"INSERT INTO {schema}.events (event_time, data) VALUES (now(), '{{}}'::jsonb) RETURNING event_id"
        )
    )


async def _seed_minimal_fact(conn, schema: str):
    event_id = await _insert_event(conn, schema)
    atom_id = await conn.fetchval(
        f"INSERT INTO {schema}.atoms (text, atom_type) VALUES ('rmt06-smoke', 'P') RETURNING atom_id"
    )
    anchor_id = await conn.fetchval(
        f"INSERT INTO {schema}.anchors (atom_id, centroid_vector) "
        f"VALUES ($1, ('[' || array_to_string(array_fill(0.1::real, ARRAY[1024]), ',') || ']')::vector) "
        f"RETURNING anchor_id",
        atom_id,
    )
    await conn.execute(
        f"INSERT INTO {schema}.event_atoms (event_id, occurrence_id, atom_id, anchor_id, role_type, target_occ) "
        f"VALUES ($1, 1, $2, $3, 'R', NULL)",
        event_id,
        atom_id,
        anchor_id,
    )
    return event_id, atom_id, anchor_id


@requires_remote
@remote_group
async def test_remote_bitmap_upserts_cardinality_and_zero_residue():
    """Group A: real rb64_build/rb64_or upserts, set idempotence, role-row
    isolation, cardinality, then DROP SCHEMA + zero-residue count proof."""
    schema = _temp_schema("nzb_rmt")
    remote = _Remote()
    tunnel = _SshTunnel()
    conn = None
    try:
        conn = await _connect(tunnel)
        production_snapshot = {table: await _table_count(conn, table) for table in _TABLES}
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            remote.run_file(_BASE_SQL_PATH, remap=schema)

            event_id, atom_id, anchor_id = await _seed_minimal_fact(conn, schema)
            plan = _plan(event_id=event_id, atom_id=atom_id, anchor_id=anchor_id)

            # First insert, then the same bits again -> set unchanged.
            await apply_bitmap_writes(conn, schema, plan)
            await apply_bitmap_writes(conn, schema, plan)
            cardinality = await conn.fetchval(
                f"SELECT rb64_cardinality(event_bitmap) FROM {schema}.cooccurrence_bitmaps "
                f"WHERE atom_id = $1 AND anchor_id = $2",
                atom_id,
                anchor_id,
            )
            assert cardinality == 1, f"repeat write changed the set: {cardinality}"
            for role in ("N", "S", "O"):
                role_cardinality = await conn.fetchval(
                    f"SELECT rb64_cardinality(neighbor_bitmap) FROM {schema}.neighbor_bitmaps "
                    f"WHERE atom_id = $1 AND anchor_id = $2 AND role_type = $3::text::\"char\"",
                    atom_id,
                    anchor_id,
                    role,
                )
                expected = {"N": 2, "S": 1, "O": 1}[role]
                assert role_cardinality == expected, f"role {role}: {role_cardinality} != {expected}"

            # Second event bit and second neighbor bit -> union grows.
            second_event = await _insert_event(conn, schema)
            second_atom = await conn.fetchval(
                f"INSERT INTO {schema}.atoms (text, atom_type) VALUES ('rmt06-smoke-2', 'E') RETURNING atom_id"
            )
            await apply_bitmap_writes(
                conn,
                schema,
                BitmapWritePlan(
                    cooccurrences=(CooccurrenceBitmapWrite(atom_id, anchor_id, second_event),),
                    neighbors=(NeighborBitmapWrite(atom_id, anchor_id, "N", (second_atom,)),),
                ),
            )
            cardinality = await conn.fetchval(
                f"SELECT rb64_cardinality(event_bitmap) FROM {schema}.cooccurrence_bitmaps "
                f"WHERE atom_id = $1 AND anchor_id = $2",
                atom_id,
                anchor_id,
            )
            assert cardinality == 2, f"second event bit lost: {cardinality}"
            neighbor_cardinality = await conn.fetchval(
                f"SELECT rb64_cardinality(neighbor_bitmap) FROM {schema}.neighbor_bitmaps "
                f"WHERE atom_id = $1 AND anchor_id = $2 AND role_type = 'N'::\"char\"",
                atom_id,
                anchor_id,
            )
            assert neighbor_cardinality == 3, f"second neighbor bit lost: {neighbor_cardinality}"
            role_rows = await conn.fetch(
                f"SELECT role_type::text AS role, rb64_cardinality(neighbor_bitmap) AS c "
                f"FROM {schema}.neighbor_bitmaps WHERE atom_id = $1 ORDER BY role_type::text",
                atom_id,
            )
            assert [(row["role"], row["c"]) for row in role_rows] == [("N", 3), ("O", 1), ("S", 1)]
        finally:
            remote.query(f"DROP SCHEMA IF EXISTS {schema} CASCADE")

        for table in _TABLES:
            assert await _table_count(conn, table) == production_snapshot[table], f"{table} leaked from the smoke"
    finally:
        if conn is not None:
            await conn.close()
        tunnel.close()
        remote.close()


@requires_remote
@remote_group
async def test_remote_bitmap_concurrent_union_no_lost_update():
    """Group B: two independent connections upsert the same rows inside
    overlapping transactions; the second ON CONFLICT waits on the first row
    lock and both bits survive (the database-side rb64_or contract)."""
    schema = _temp_schema("nzb2_rmt")
    remote = _Remote()
    tunnel = _SshTunnel()
    conn = conn_a = conn_b = None
    try:
        conn = await _connect(tunnel)
        production_snapshot = {table: await _table_count(conn, table) for table in _TABLES}
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            remote.run_file(_BASE_SQL_PATH, remap=schema)
            atom_id = await conn.fetchval(
                f"INSERT INTO {schema}.atoms (text, atom_type) VALUES ('rmt06-concurrent', 'E') RETURNING atom_id"
            )
            anchor_id = 900001

            conn_a = await _connect(tunnel)
            conn_b = await _connect(tunnel)

            async def writer(connection, event_id, neighbor_bit):
                async with connection.transaction():
                    await apply_bitmap_writes(
                        connection,
                        schema,
                        BitmapWritePlan(
                            cooccurrences=(CooccurrenceBitmapWrite(atom_id, anchor_id, event_id),),
                            neighbors=(NeighborBitmapWrite(atom_id, anchor_id, "N", (neighbor_bit,)),),
                        ),
                    )
                    await asyncio.sleep(0.3)

            await asyncio.gather(writer(conn_a, 901, 900000011), writer(conn_b, 902, 900000012))

            event_cardinality = await conn.fetchval(
                f"SELECT rb64_cardinality(event_bitmap) FROM {schema}.cooccurrence_bitmaps "
                f"WHERE atom_id = $1 AND anchor_id = $2",
                atom_id,
                anchor_id,
            )
            neighbor_cardinality = await conn.fetchval(
                f"SELECT rb64_cardinality(neighbor_bitmap) FROM {schema}.neighbor_bitmaps "
                f"WHERE atom_id = $1 AND anchor_id = $2 AND role_type = 'N'::\"char\"",
                atom_id,
                anchor_id,
            )
            assert event_cardinality == 2, f"lost update on event bitmap: {event_cardinality}"
            assert neighbor_cardinality == 2, f"lost update on neighbor bitmap: {neighbor_cardinality}"
            cooccurrence_rows = await conn.fetchval(
                f"SELECT count(*) FROM {schema}.cooccurrence_bitmaps WHERE atom_id = $1 AND anchor_id = $2",
                atom_id,
                anchor_id,
            )
            assert cooccurrence_rows == 1, "concurrent inserts duplicated the row"
        finally:
            remote.query(f"DROP SCHEMA IF EXISTS {schema} CASCADE")

        for table in _TABLES:
            assert await _table_count(conn, table) == production_snapshot[table], f"{table} leaked from the smoke"
    finally:
        for connection in (conn_a, conn_b, conn):
            if connection is not None:
                await connection.close()
        tunnel.close()
        remote.close()
