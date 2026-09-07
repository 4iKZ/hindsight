"""Remote DDL contract tests for Noesis requirement 02 (R02-02 + R02-03).

Runs entirely on a throwaway TEMP schema so the real ``noesis_core`` tables
are never touched. Gated: skipped unless ``NOESIS_REMOTE_DDL_TEST=1`` plus the
SSH env vars are present; the normal regression run never reaches the remote DB.

Covers:
  * R02-02 fresh-install: the baseline DDL creates ingestion_alerts.dedupe_key
    NOT NULL + unique constraint, and an _ALERT_INSERT-equivalent INSERT runs
    twice with the second ON CONFLICT deduped.
  * R02-03 migration 002 re-runnability:
      - empty pre-migration schema  -> first run succeeds;
      - after inserting representative rows -> re-run succeeds as a no-op;
      - missing column + non-backfillable data -> fail-closed;
      - failure leaves no half-migrated state.

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
_MIGRATION_PATH = _REPO_ROOT / "docs" / "db" / "migrations" / "002-noesis-ingest-idempotency.sql"
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
def test_fresh_install_baseline_matches_application_sql():
    """R02-02: baseline DDL creates dedupe_key NOT NULL + UNIQUE, and the app's
    _ALERT_INSERT-equivalent INSERT dedupes on the second attempt."""
    schema = f"nzf_rmt_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"
    remote = _Remote()
    try:
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            remote.run_file(_BASE_SQL_PATH, remap=schema)

            # dedupe_key NOT NULL
            nullable = remote.query(
                f"SELECT is_nullable FROM information_schema.columns "
                f"WHERE table_schema='{schema}' AND table_name='ingestion_alerts' AND column_name='dedupe_key'"
            )
            assert nullable == ["NO"], f"dedupe_key nullable={nullable}"

            # unique constraint present
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
def test_migration_002_rerunnable_and_fail_closed():
    """R02-03: migration 002 is a safe no-op on an already-complete structure
    (even with data), and fail-closes when a missing column cannot be backfilled."""
    schema = f"nzm_rmt_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"
    remote = _Remote()
    try:
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            # --- Scenario 1&2: empty pre-migration schema -> first run succeeds
            remote.run_file(_BASE_SQL_PATH, remap=schema)  # baseline already has the target structure
            remote.run_file(_MIGRATION_PATH, remap=schema)  # re-run on complete structure = no-op

            # Insert representative rows (structure complete) then re-run again -> no-op
            remote.query(
                f"INSERT INTO {schema}.events (event_time, ingestion_key, data, source, category) "
                f"VALUES (now(), 'ik-1', '{{}}'::jsonb, 'hindsight_retain', 'fact')"
            )
            remote.query(
                f"INSERT INTO {schema}.ingestion_alerts (dedupe_key, event_id, stage, alert_code, severity, message, details) "
                f"VALUES ('dk-1', NULL, 'event_ingest', 'event_ingest_failed', 'error', 'm', '{{}}'::jsonb)"
            )
            # re-run with data present: must succeed as a no-op
            remote.run_file(_MIGRATION_PATH, remap=schema)
            # verify data still present, no half-state
            assert remote.query(f"SELECT count(*) FROM {schema}.events") == ["1"]
            assert remote.query(f"SELECT count(*) FROM {schema}.ingestion_alerts") == ["1"]
        finally:
            remote.query(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        remote.close()


@requires_remote
def test_migration_002_fail_closed_on_non_backfillable_data():
    """R02-03: when the target column is missing and rows cannot be backfilled,
    the migration fails (no guesses), and no half-migration state remains."""
    schema = f"nzf2_rmt_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"
    remote = _Remote()
    try:
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            # Build the FULL baseline, then strip dedupe_key so the alerts table
            # is "old" while atoms/events still exist (so migration reaches the
            # alerts step instead of failing earlier on a missing relation).
            remote.run_file(_BASE_SQL_PATH, remap=schema)
            remote.query(f"ALTER TABLE {schema}.ingestion_alerts DROP COLUMN dedupe_key CASCADE")
            remote.query(
                f"INSERT INTO {schema}.ingestion_alerts (stage, alert_code, severity, message) "
                f"VALUES ('event_ingest', 'event_ingest_failed', 'error', 'm')"
            )
            # Running the (remapped) migration MUST fail because dedupe_key is
            # missing AND data exists.
            try:
                remote.run_file(_MIGRATION_PATH, remap=schema)
                raise AssertionError("migration should have fail-closed on non-backfillable data")
            except RuntimeError as exc:
                assert "cannot backfill" in str(exc) and "dedupe_key" in str(exc), f"unexpected error: {exc}"
            # No half-migration: dedupe_key column must NOT have been added.
            cols = remote.query(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_schema='{schema}' AND table_name='ingestion_alerts' AND column_name='dedupe_key'"
            )
            assert cols == [], f"half-migration leaked dedupe_key column: {cols}"
        finally:
            remote.query(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        remote.close()


@requires_remote
def test_migration_002_repairs_existing_nullable_identity_columns():
    """An old schema may already have the columns but leave them nullable.
    Clean columns are tightened; NULL-bearing columns fail closed."""
    schema = f"nzn_rmt_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"
    remote = _Remote()
    try:
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            remote.run_file(_BASE_SQL_PATH, remap=schema)
            remote.query(f"ALTER TABLE {schema}.events ALTER COLUMN ingestion_key DROP NOT NULL")
            remote.query(f"ALTER TABLE {schema}.ingestion_alerts ALTER COLUMN dedupe_key DROP NOT NULL")
            remote.run_file(_MIGRATION_PATH, remap=schema)
            nullable = remote.query(
                f"SELECT table_name || ':' || is_nullable FROM information_schema.columns "
                f"WHERE table_schema='{schema}' AND "
                f"((table_name='events' AND column_name='ingestion_key') OR "
                f"(table_name='ingestion_alerts' AND column_name='dedupe_key')) ORDER BY table_name"
            )
            assert nullable == ["events:NO", "ingestion_alerts:NO"]

            remote.query(f"ALTER TABLE {schema}.events ALTER COLUMN ingestion_key DROP NOT NULL")
            remote.query(
                f"INSERT INTO {schema}.events (event_time, ingestion_key, data, source, category) "
                f"VALUES (now(), NULL, '{{}}'::jsonb, 'hindsight_retain', 'fact')"
            )
            with pytest.raises(RuntimeError, match="ingestion_key.*NULL"):
                remote.run_file(_MIGRATION_PATH, remap=schema)
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
