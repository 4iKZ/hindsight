"""Remote DDL contract tests for the Noesis 04A latest-contract schema.

Runs entirely on a throwaway TEMP schema so the real ``noesis_core`` tables
are never touched. Gated: skipped unless ``NOESIS_REMOTE_DDL_TEST=1`` plus the
SSH env vars are present; the normal regression run never reaches the remote DB.

Covers (04A requirement §6 / §11.2):
  * fresh-install baseline: atoms carry no support_count; anchors carry no
    merged_into_anchor_id; events carry no ingestion_key (sequence-numbered,
    no unique constraint, plain idx_events_event_id); event_atoms carries
    target_occ and no head column; class_membership primary key is
    (atom_id, anchor_id); role_type / source / category / status use "char"
    CHECK enums (A/P/R/M, L/A/P/S/M, R/W/C/G, A/D).
  * ingestion_alerts.dedupe_key NOT NULL + unique constraint, and an
    _ALERT_INSERT-equivalent INSERT dedupes on the second attempt.
  * migration 003 is idempotent and only creates the identity profile gate.

Credentials live only in the environment; nothing is written to source, logs,
or fixtures. The remote host key is pinned.
"""

from __future__ import annotations

import base64
import hashlib
import os
import random
import string
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

pytest.importorskip("paramiko")

_EXPECTED_HOST_KEY = "SHA256:fYPgM4a2OY1ZRhdQbx2z2YjiQ9bOMx4zo/c1ewn+WCs"
_REPO_ROOT = Path(__file__).resolve().parents[4]  # wxs-noesis/
_BASE_SQL_PATH = _REPO_ROOT / "docs" / "db" / "noesis-stage1-schema.sql"
_MIGRATION_003_PATH = _REPO_ROOT / "docs" / "db" / "migrations" / "003-noesis-identity-embedding-profile.sql"


def _enabled() -> bool:
    return (
        os.environ.get("NOESIS_REMOTE_DDL_TEST") == "1"
        and all(os.environ.get(k) for k in ("NOESIS_SSH_HOST", "NOESIS_SSH_PORT", "NOESIS_SSH_USER", "NOESIS_SSH_PW"))
    )


requires_remote = pytest.mark.skipif(not _enabled(), reason="NOESIS_REMOTE_DDL_TEST/SSH env not set")


class _Remote:
    """SSH host-key-pinned helper that uploads a script and runs psql -f."""

    def __init__(self) -> None:
        import paramiko

        self.transport = paramiko.Transport((os.environ["NOESIS_SSH_HOST"], int(os.environ["NOESIS_SSH_PORT"])))
        self.transport.start_client(timeout=20)
        key = self.transport.get_remote_server_key()
        fp = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        if fp != _EXPECTED_HOST_KEY:
            self.transport.close()
            raise RuntimeError("remote host key mismatch — refusing to connect")
        self.transport.auth_password(username=os.environ["NOESIS_SSH_USER"], password=os.environ["NOESIS_SSH_PW"])
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def run_file(self, local_path: str, *, remap: str | None = None) -> str:
        """Upload ``local_path`` (optionally remapping ``noesis_core`` -> ``remap``) and psql -f it."""
        content = open(local_path, encoding="utf-8").read()
        if remap:
            # Whole-token remap so both dotted refs and string literals like
            # table_schema='noesis_core' land on the temp schema. The
            # CREATE SCHEMA IF NOT EXISTS ... stays idempotent.
            content = content.replace("noesis_core", remap)
        remote = f"/tmp/r02_ddl_{uuid.uuid4().hex}.sql"
        with self.sftp.open(remote, "w") as fh:
            fh.write(content)
        chan = self.transport.open_session()
        chan.settimeout(60)
        chan.exec_command(
            f"/usr/local/pgsql18/bin/psql -U postgres -d noesis -v ON_ERROR_STOP=1 -f {remote} "
            f"&& rm -f {remote} && echo MIGRATION_OK"
        )
        chan.makefile().read()
        err = chan.makefile_stderr().read().decode().strip()
        rc = chan.recv_exit_status()
        chan.close()
        if rc != 0:
            raise RuntimeError(f"psql failed (rc={rc}): {err}")
        return "OK"

    def query(self, sql: str) -> list[str]:
        chan = self.transport.open_session()
        chan.settimeout(30)
        chan.exec_command(
            "/usr/local/pgsql18/bin/psql -U postgres -d noesis -At -c \"" + sql.replace('"', '\\"') + "\""
        )
        out = chan.makefile().read().decode().strip()
        err = chan.makefile_stderr().read().decode().strip()
        rc = chan.recv_exit_status()
        chan.close()
        if rc != 0:
            raise RuntimeError(f"psql query failed (rc={rc}): {err}")
        return [line for line in out.splitlines() if line]

    def close(self) -> None:
        self.sftp.close()
        self.transport.close()


@requires_remote
def test_fresh_install_latest_contract_catalog():
    """04A §6: the baseline DDL creates exactly the latest-contract structure.

    Catalog proof for: atoms (no support_count, "char" enums), anchors (no
    merged_into, E/P shared), events (no ingestion_key, no unique constraint,
    plain idx_events_event_id), event_atoms (target_occ, no head column,
    A/P/R/M), class_membership ((atom_id, anchor_id) primary key)."""
    schema = f"nzf_rmt_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"
    remote = _Remote()
    try:
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            remote.run_file(_BASE_SQL_PATH, remap=schema)

            def columns(table: str) -> list[str]:
                return remote.query(
                    f"SELECT column_name FROM information_schema.columns "
                    f"WHERE table_schema='{schema}' AND table_name='{table}' ORDER BY ordinal_position"
                )

            # atoms: no support_count; text/atom_type/embedding/status present.
            atom_columns = columns("atoms")
            assert "support_count" not in atom_columns, f"atoms leaked support_count: {atom_columns}"
            for required in ("text", "atom_type", "embedding", "status"):
                assert required in atom_columns, f"atoms.{required} missing: {atom_columns}"

            # anchors: no merged_into_anchor_id.
            anchor_columns = columns("anchors")
            assert "merged_into_anchor_id" not in anchor_columns, f"anchors leaked merged_into: {anchor_columns}"

            # events: no ingestion_key, no unique constraint, plain event_id index.
            event_columns = columns("events")
            assert "ingestion_key" not in event_columns, f"events leaked ingestion_key: {event_columns}"
            for required in ("event_id", "event_time", "data", "source", "category"):
                assert required in event_columns, f"events.{required} missing: {event_columns}"
            event_uniques = remote.query(
                f"SELECT conname FROM pg_constraint WHERE conrelid='{schema}.events'::regclass "
                f"AND contype='u'"
            )
            assert event_uniques == [], f"events must carry no unique constraint: {event_uniques}"
            event_indexes = remote.query(
                f"SELECT indexname FROM pg_indexes WHERE schemaname='{schema}' AND tablename='events'"
            )
            assert "idx_events_event_id" in event_indexes, f"idx_events_event_id missing: {event_indexes}"

            # event_atoms: target_occ present, head column absent.
            event_atom_columns = columns("event_atoms")
            assert "target_occ" in event_atom_columns, f"event_atoms.target_occ missing: {event_atom_columns}"
            assert "head_occurrence_id" not in event_atom_columns, (
                f"event_atoms leaked head_occurrence_id: {event_atom_columns}"
            )

            # class_membership: composite primary key (atom_id, anchor_id).
            membership_pk = remote.query(
                f"SELECT a.attname FROM pg_index i "
                f"JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                f"WHERE i.indrelid='{schema}.class_membership'::regclass AND i.indisprimary "
                f"ORDER BY a.attnum"
            )
            assert membership_pk == ["atom_id", "anchor_id"], f"class_membership pk wrong: {membership_pk}"

            # "char" CHECK enums: role_type A/P/R/M, source, category, status.
            def check_defs(column: str, table: str) -> str:
                defs = remote.query(
                    f"SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                    f"WHERE conrelid='{schema}.{table}'::regclass AND contype='c' "
                    f"AND pg_get_constraintdef(oid) ILIKE '%{column}%'"
                )
                assert defs, f"no CHECK definition for {table}.{column}"
                return defs[0]

            assert "role_type" in check_defs("role_type", "event_atoms")
            for value in ("'A'", "'P'", "'R'", "'M'"):
                assert value in check_defs("role_type", "event_atoms"), f"role_type missing {value}"
            for value in ("'L'", "'A'", "'P'", "'S'", "'M'"):
                assert value in check_defs("source", "events"), f"source missing {value}"
            for value in ("'R'", "'W'", "'C'", "'G'"):
                assert value in check_defs("category", "events"), f"category missing {value}"
            for value in ("'A'", "'D'"):
                assert value in check_defs("status", "atoms"), f"atoms.status missing {value}"

            # ingestion_alerts: dedupe_key NOT NULL + unique (contract unchanged).
            nullable = remote.query(
                f"SELECT is_nullable FROM information_schema.columns "
                f"WHERE table_schema='{schema}' AND table_name='ingestion_alerts' AND column_name='dedupe_key'"
            )
            assert nullable == ["NO"], f"dedupe_key nullable={nullable}"
            con = remote.query(
                f"SELECT conname FROM pg_constraint WHERE conrelid='{schema}.ingestion_alerts'::regclass "
                f"AND conname='ingestion_alerts_dedupe_key_unique'"
            )
            assert con == ["ingestion_alerts_dedupe_key_unique"], f"missing unique constraint: {con}"

            # _ALERT_INSERT-equivalent twice -> second on-conflict is deduped
            ins = (
                f"INSERT INTO {schema}.ingestion_alerts (dedupe_key, event_id, stage, alert_code, severity, message, details, status) "
                f"VALUES ('dedupe-key-1', NULL, 'event_ingest', 'event_ingest_failed', 'error', 'm', '{{}}'::jsonb, 'open') "
                f"ON CONFLICT (dedupe_key) DO NOTHING"
            )
            remote.query(ins)
            remote.query(ins)
            cnt = remote.query(f"SELECT count(*) FROM {schema}.ingestion_alerts WHERE dedupe_key='dedupe-key-1'")
            assert cnt == ["1"], f"alert dedupe failed: {cnt}"
        finally:
            remote.query(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        remote.close()


@requires_remote
def test_migration_003_idempotent_creates_identity_profile_gate():
    """Req-03 §11: migration 003 is idempotent and only creates the single-row
    model-generation gate — never an index on atoms.embedding (IVFFlat deferred
    to req-04), never a fabricated profile row, and no changes to fact tables."""
    schema = f"nz3_rmt_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"
    remote = _Remote()
    try:
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            # Baseline must already expose atoms.embedding VECTOR(1024).
            remote.run_file(_BASE_SQL_PATH, remap=schema)

            # Run migration 003 twice: both succeed (idempotent).
            remote.run_file(_MIGRATION_003_PATH, remap=schema)
            remote.run_file(_MIGRATION_003_PATH, remap=schema)

            # Table exists with the frozen columns.
            cols = remote.query(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_schema='{schema}' AND table_name='embedding_profiles' ORDER BY ordinal_position"
            )
            assert cols == [
                "embedding_kind",
                "model_name",
                "model_revision",
                "dimension",
                "status",
                "updated_at",
            ], cols

            # No profile row fabricated by the migration (online claim only).
            rows = remote.query(f"SELECT count(*) FROM {schema}.embedding_profiles")
            assert rows == ["0"], f"migration fabricated a profile row: {rows}"

            # No IVFFlat/HNSW index on atoms.embedding was created.
            indexes = remote.query(
                f"SELECT indexdef FROM pg_indexes WHERE schemaname='{schema}' AND tablename='atoms' "
                f"AND indexdef ILIKE '%embedding%'"
            )
            assert indexes == [], f"req-04 index leaked by migration 003: {indexes}"
        finally:
            remote.query(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        remote.close()
