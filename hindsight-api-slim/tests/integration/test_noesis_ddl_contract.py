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

Covers (requirement 05 §10.2/§13.6, offline text + gated remote behavior):
  * the canonical DDL text pins event_atoms.anchor_id as BIGINT NOT NULL
    REFERENCES noesis_core.anchors(anchor_id) and stays free of
    support_count / merged_into_anchor_id / redirect constructs;
  * migration 005 aborts on existing NULL anchor rows with the exact count
    and leaves the data untouched, then applies the idempotent SET NOT NULL
    once the rows are resolved; re-running is safe; its text carries no
    DELETE/TRUNCATE/INSERT/UPDATE business DML.

Credentials live only in the environment; nothing is written to source, logs,
or fixtures. The remote host key is pinned.
"""

from __future__ import annotations

import base64
import hashlib
import os
import random
import re
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
_MIGRATION_005_PATH = _REPO_ROOT / "docs" / "db" / "migrations" / "005-noesis-anchor-routing.sql"
_MIGRATION_006_PATH = (
    _REPO_ROOT / "docs" / "db" / "migrations" / "006-noesis-shared-embedding-generation.sql"
)
_MIGRATION_007_PATH = (
    _REPO_ROOT / "docs" / "db" / "migrations" / "007-noesis-bitmap-write-maintenance.sql"
)


def _enabled() -> bool:
    return (
        os.environ.get("NOESIS_REMOTE_DDL_TEST") == "1"
        and all(os.environ.get(k) for k in ("NOESIS_SSH_HOST", "NOESIS_SSH_PORT", "NOESIS_SSH_USER", "NOESIS_SSH_PW"))
    )


requires_remote = pytest.mark.skipif(not _enabled(), reason="NOESIS_REMOTE_DDL_TEST/SSH env not set")

# Serialize with every other remote noesis test under pytest-xdist (see
# test_noesis_remote_pg.remote_group for the why).
remote_group = pytest.mark.xdist_group("noesis_remote")


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
@remote_group
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

            # Requirement 06 §8.1: metric-layer bitmap tables use real anchors.
            def pk_columns(table: str) -> list[str]:
                return remote.query(
                    f"SELECT a.attname FROM pg_index i "
                    f"JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                    f"WHERE i.indrelid='{schema}.{table}'::regclass AND i.indisprimary "
                    f"ORDER BY a.attnum"
                )

            assert columns("cooccurrence_bitmaps") == ["atom_id", "anchor_id", "event_bitmap"]
            assert columns("neighbor_bitmaps") == [
                "atom_id",
                "anchor_id",
                "role_type",
                "neighbor_bitmap",
            ]
            assert pk_columns("cooccurrence_bitmaps") == ["atom_id", "anchor_id"]
            assert pk_columns("neighbor_bitmaps") == ["atom_id", "anchor_id", "role_type"]
            for table in ("cooccurrence_bitmaps", "neighbor_bitmaps"):
                anchor_default = remote.query(
                    f"SELECT column_default FROM information_schema.columns "
                    f"WHERE table_schema='{schema}' AND table_name='{table}' AND column_name='anchor_id'"
                )
                assert anchor_default == [], f"{table}.anchor_id must not carry a DEFAULT: {anchor_default}"
                anchor_checks = remote.query(
                    f"SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                    f"WHERE conrelid='{schema}.{table}'::regclass AND contype='c' "
                    f"AND pg_get_constraintdef(oid) ILIKE '%anchor_id%'"
                )
                assert anchor_checks and "anchor_id > 0" in anchor_checks[0], anchor_checks
            role_checks = remote.query(
                f"SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                f"WHERE conrelid='{schema}.neighbor_bitmaps'::regclass AND contype='c' "
                f"AND pg_get_constraintdef(oid) ILIKE '%role_type%'"
            )
            assert role_checks and all(value in role_checks[0] for value in ("'S'", "'O'", "'N'")), role_checks
            bitmap_types = remote.query(
                f"SELECT a.attname || ':' || format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                f"WHERE a.attrelid IN ('{schema}.cooccurrence_bitmaps'::regclass, "
                f"'{schema}.neighbor_bitmaps'::regclass) "
                f"AND a.attname IN ('event_bitmap', 'neighbor_bitmap') ORDER BY 1"
            )
            assert bitmap_types == ["event_bitmap:roaringbitmap64", "neighbor_bitmap:roaringbitmap64"], bitmap_types
            bitmap_fks = remote.query(
                f"SELECT conrelid::regclass::text FROM pg_constraint "
                f"WHERE contype='f' AND confrelid='{schema}.atoms'::regclass "
                f"AND conrelid IN ('{schema}.cooccurrence_bitmaps'::regclass, '{schema}.neighbor_bitmaps'::regclass) "
                f"ORDER BY 1"
            )
            assert len(bitmap_fks) == 2, f"bitmap atom FKs missing: {bitmap_fks}"
            neighbor_indexes = remote.query(
                f"SELECT indexname FROM pg_indexes WHERE schemaname='{schema}' AND tablename='neighbor_bitmaps'"
            )
            assert "idx_neighbor_atom_anchor" in neighbor_indexes, neighbor_indexes

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
@remote_group
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


@requires_remote
@remote_group
def test_migration_005_rejects_null_anchors_then_applies_idempotently():
    """Requirement 05 §10.2: migration 005 aborts with the exact NULL count
    while legacy NULL anchor rows exist (data untouched, column still
    nullable), then applies the idempotent SET NOT NULL once the rows are
    resolved; a re-run is a safe no-op."""
    schema = f"nz5_rmt_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"
    remote = _Remote()
    try:
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            # Fresh baseline, then simulate a pre-005 deployment: anchor_id
            # still nullable with one legacy NULL row.
            remote.run_file(_BASE_SQL_PATH, remap=schema)
            remote.query(f"ALTER TABLE {schema}.event_atoms ALTER COLUMN anchor_id DROP NOT NULL")
            event_id = remote.query(
                f"INSERT INTO {schema}.events (event_time, data) VALUES (now(), '{{}}'::jsonb) RETURNING event_id"
            )[0]
            atom_id = remote.query(
                f"INSERT INTO {schema}.atoms (text, atom_type) VALUES ('apple', 'E') RETURNING atom_id"
            )[0]
            remote.query(
                f"INSERT INTO {schema}.event_atoms (event_id, occurrence_id, atom_id, anchor_id, role_type, "
                f"target_occ) VALUES ({event_id}, 1, {atom_id}, NULL, 'A', NULL)"
            )

            def anchor_id_nullable() -> list[str]:
                return remote.query(
                    f"SELECT is_nullable FROM information_schema.columns "
                    f"WHERE table_schema='{schema}' AND table_name='event_atoms' AND column_name='anchor_id'"
                )

            # 005 must abort on the NULL row and leave it exactly as-is.
            with pytest.raises(RuntimeError, match="anchor_id IS NULL"):
                remote.run_file(_MIGRATION_005_PATH, remap=schema)
            null_rows = remote.query(
                f"SELECT count(*) FROM {schema}.event_atoms WHERE anchor_id IS NULL"
            )
            assert null_rows == ["1"], f"NULL row was modified by the aborted migration: {null_rows}"
            assert anchor_id_nullable() == ["YES"], "aborted migration still flipped NOT NULL"

            # Leader resolves the row; 005 applies and is safe to re-run.
            remote.query(f"DELETE FROM {schema}.event_atoms WHERE anchor_id IS NULL")
            remote.run_file(_MIGRATION_005_PATH, remap=schema)
            remote.run_file(_MIGRATION_005_PATH, remap=schema)
            assert anchor_id_nullable() == ["NO"], "SET NOT NULL not applied after NULL rows were resolved"
        finally:
            remote.query(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        remote.close()


# ---------------------------------------------------------------------------
# Offline text-contract assertions (no remote access required)
# ---------------------------------------------------------------------------

def _strip_sql_comments(text: str) -> str:
    """Drop full-line SQL comments so text assertions see executable code."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("--"))


def _table_block(text: str, table: str) -> str:
    start = text.index(f"CREATE TABLE noesis_core.{table}")
    return text[start : text.index(");", start)]


def test_canonical_ddl_event_atoms_anchor_id_is_not_null_fk():
    """Requirement 05 §10.1: event_atoms.anchor_id is BIGINT NOT NULL
    referencing noesis_core.anchors(anchor_id) — no NULL default bucket."""
    text = _BASE_SQL_PATH.read_text(encoding="utf-8")
    block = _table_block(text, "event_atoms")
    assert re.search(
        r"anchor_id\s+BIGINT\s+NOT\s+NULL\s+REFERENCES\s+noesis_core\.anchors\(anchor_id\)",
        block,
    ), "event_atoms.anchor_id must be BIGINT NOT NULL REFERENCES noesis_core.anchors(anchor_id)"


def test_canonical_ddl_has_no_support_merge_redirect_constructs():
    """04A/05: support_count, merged_into_anchor_id, and any redirect
    replacement stay banned from the executable DDL (comments may explain)."""
    code = _strip_sql_comments(_BASE_SQL_PATH.read_text(encoding="utf-8"))
    for banned in ("support_count", "merged_into_anchor_id", "redirect"):
        assert banned not in code, f"canonical DDL leaked banned construct {banned!r}"


def test_canonical_ddl_bitmap_tables_contract():
    """Requirement 06 §8.1: real non-zero anchors, no DEFAULT 0, role_type in
    the neighbor primary key, canonical atom FKs, and the lookup index."""
    code = _strip_sql_comments(_BASE_SQL_PATH.read_text(encoding="utf-8"))
    for table, bitmap_column in (("cooccurrence_bitmaps", "event_bitmap"), ("neighbor_bitmaps", "neighbor_bitmap")):
        block = _table_block(code, table)
        assert "DEFAULT 0" not in block, f"{table}.anchor_id must not carry a DEFAULT 0"
        assert re.search(
            r"anchor_id\s+BIGINT\s+NOT\s+NULL\s+CHECK\s*\(\s*anchor_id\s*>\s*0\s*\)", block
        ), f"{table}.anchor_id must be BIGINT NOT NULL CHECK (anchor_id > 0)"
        assert re.search(
            r"atom_id\s+BIGINT\s+NOT\s+NULL\s+REFERENCES\s+noesis_core\.atoms\(atom_id\)\s+ON DELETE CASCADE",
            block,
        ), f"{table}.atom_id must be NOT NULL with the canonical atom FK"
        assert re.search(bitmap_column + r"\s+ROARINGBITMAP64\s+NOT\s+NULL", block, re.IGNORECASE), (
            f"{table}.{bitmap_column} must be ROARINGBITMAP64 NOT NULL"
        )
    neighbor = _table_block(code, "neighbor_bitmaps")
    assert 'role_type "char" NOT NULL' in neighbor, "neighbor_bitmaps.role_type must be a NOT NULL \"char\""
    assert re.search(
        r"role_type\s+IN\s*\(\s*'S'\s*,\s*'O'\s*,\s*'N'\s*\)", neighbor
    ), "neighbor role_type CHECK must allow exactly S/O/N"
    assert "PRIMARY KEY (atom_id, anchor_id, role_type)" in neighbor, (
        "neighbor_bitmaps PK must include role_type"
    )
    assert "CREATE INDEX idx_neighbor_atom_anchor" in code, "idx_neighbor_atom_anchor is missing"


def test_migration_005_text_contract():
    """Requirement 05 §10.2: migration 005 aborts on existing NULL anchor
    rows (RAISE with the exact count for the Leader), applies SET NOT NULL
    only behind the nullable guard, and carries no business DML."""
    code = _strip_sql_comments(_MIGRATION_005_PATH.read_text(encoding="utf-8"))
    # Stock check: refuse to run while NULL anchor rows exist.
    assert "WHERE anchor_id IS NULL" in code, "migration 005 lost the NULL stock check"
    assert "RAISE EXCEPTION" in code, "migration 005 must RAISE on NULL anchor rows"
    # Idempotent SET NOT NULL guarded by the nullable probe.
    assert "is_nullable = 'YES'" in code, "migration 005 lost the idempotent nullable guard"
    assert "SET NOT NULL" in code, "migration 005 lost the SET NOT NULL step"
    # Never any business DML: no backfill, no destructive cleanup.
    for banned in ("INSERT", "UPDATE", "DELETE", "TRUNCATE"):
        assert banned not in code, f"migration 005 leaked business DML: {banned}"


def test_migration_006_text_contract():
    """Requirement 05A §8.2: migration 006 is catalog guard + COMMENT only —
    it never writes the profile row, never touches business data, and never
    creates or drops an ANN index."""
    code = _strip_sql_comments(_MIGRATION_006_PATH.read_text(encoding="utf-8"))
    # Fail-closed identity + pre-state guards.
    assert "current_database()" in code, "migration 006 lost the database guard"
    assert "RAISE EXCEPTION" in code, "migration 006 must raise on a mismatched pre-state"
    assert "vector(1024)" in code, "migration 006 lost the 1024-dimension guard"
    # Comment alignment for the shared E/P embedding space.
    assert "COMMENT ON TABLE" in code, "migration 006 lost the table comment"
    assert "shared" in code.lower(), "migration 006 comment must describe the shared embedding space"
    # Never any business DML, and never an ANN index change.
    for banned in ("INSERT", "UPDATE", "DELETE", "TRUNCATE"):
        assert banned not in code, f"migration 006 leaked business DML: {banned}"
    for banned in ("CREATE INDEX", "DROP INDEX", "REINDEX", "CONCURRENTLY"):
        assert banned not in code, f"migration 006 must not touch an ANN index: {banned}"


def test_migration_007_text_contract():
    """Requirement 06 §8.2: migration 007 locks the two bitmap tables, fails
    closed on anchor_id=0/NULL and on a non-empty legacy neighbor table without
    role_type, drops the DEFAULT 0, tightens NOT NULL, rebuilds the neighbor
    PK with role_type, adds the canonical atom FKs, and never runs business
    DML or destructive cleanup."""
    code = _strip_sql_comments(_MIGRATION_007_PATH.read_text(encoding="utf-8"))
    assert "current_database()" in code, "migration 007 lost the database guard"
    assert "LOCK TABLE" in code, "migration 007 must lock the two bitmap tables"
    assert "RAISE EXCEPTION" in code, "migration 007 must fail closed on legacy data"
    assert "anchor_id = 0" in code, "migration 007 lost the anchor_id = 0 stock check"
    assert "anchor_id IS NULL" in code, "migration 007 lost the anchor_id IS NULL stock check"
    assert "role_type" in code, "migration 007 must add/protect neighbor role_type"
    assert "SET NOT NULL" in code, "migration 007 must tighten anchor_id/atom_id to NOT NULL"
    assert "DROP DEFAULT" in code, "migration 007 must drop the anchor_id DEFAULT 0"
    assert "ADD PRIMARY KEY (atom_id, anchor_id, role_type)" in code, (
        "migration 007 must rebuild the neighbor primary key with role_type"
    )
    assert "CREATE INDEX IF NOT EXISTS idx_neighbor_atom_anchor" in code, (
        "migration 007 must create the neighbor lookup index idempotently"
    )
    assert "ON DELETE CASCADE" in code, "migration 007 must add the canonical atom FKs"
    for banned in (r"\bINSERT\s+INTO\b", r"\bDELETE\s+FROM\b", r"\bTRUNCATE\b", r"\bUPDATE\b"):
        assert not re.search(banned, code, re.IGNORECASE), f"migration 007 leaked business DML: {banned}"


@requires_remote
@remote_group
def test_migration_007_upgrades_empty_legacy_bitmap_tables_idempotently():
    """Requirement 06 §12.6: the pre-007 empty bitmap shape upgrades to the
    real-anchor shape, and the migration is safe to re-run."""
    schema = f"nz7_rmt_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"
    remote = _Remote()
    try:
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            remote.run_file(_BASE_SQL_PATH, remap=schema)
            # Fresh baseline is already 007-shaped: the migration must be a no-op.
            remote.run_file(_MIGRATION_007_PATH, remap=schema)
            remote.run_file(_MIGRATION_007_PATH, remap=schema)

            # Simulate the pre-007 deployment shape.
            remote.query(f"DROP TABLE {schema}.neighbor_bitmaps, {schema}.cooccurrence_bitmaps")
            remote.query(
                f"CREATE TABLE {schema}.cooccurrence_bitmaps ("
                f"atom_id BIGINT, anchor_id BIGINT DEFAULT 0, "
                f"event_bitmap ROARINGBITMAP64 NOT NULL, PRIMARY KEY (atom_id, anchor_id))"
            )
            remote.query(
                f"CREATE TABLE {schema}.neighbor_bitmaps ("
                f"atom_id BIGINT, anchor_id BIGINT DEFAULT 0, "
                f"neighbor_bitmap ROARINGBITMAP64 NOT NULL, PRIMARY KEY (atom_id, anchor_id))"
            )
            remote.run_file(_MIGRATION_007_PATH, remap=schema)
            remote.run_file(_MIGRATION_007_PATH, remap=schema)

            cols = remote.query(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_schema='{schema}' AND table_name='neighbor_bitmaps' ORDER BY ordinal_position"
            )
            # ALTER TABLE ADD COLUMN appends role_type after neighbor_bitmap on
            # an upgraded table; the fresh-install ordinal order is not part of
            # the contract, the column set and keys are.
            assert cols and sorted(cols) == ["anchor_id", "atom_id", "neighbor_bitmap", "role_type"], cols
            cooccurrence_default = remote.query(
                f"SELECT column_default FROM information_schema.columns "
                f"WHERE table_schema='{schema}' AND table_name='cooccurrence_bitmaps' AND column_name='anchor_id'"
            )
            assert cooccurrence_default == [], cooccurrence_default
            neighbor_pk = remote.query(
                f"SELECT a.attname FROM pg_index i "
                f"JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                f"WHERE i.indrelid='{schema}.neighbor_bitmaps'::regclass AND i.indisprimary ORDER BY a.attnum"
            )
            assert neighbor_pk == ["atom_id", "anchor_id", "role_type"], neighbor_pk
            indexes = remote.query(
                f"SELECT indexname FROM pg_indexes WHERE schemaname='{schema}' AND tablename='neighbor_bitmaps'"
            )
            assert "idx_neighbor_atom_anchor" in indexes, indexes
            anchor_fks = remote.query(
                f"SELECT count(*) FROM pg_constraint WHERE contype='f' "
                f"AND confrelid='{schema}.atoms'::regclass "
                f"AND conrelid IN ('{schema}.cooccurrence_bitmaps'::regclass, '{schema}.neighbor_bitmaps'::regclass)"
            )
            assert anchor_fks == ["2"], anchor_fks
        finally:
            remote.query(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        remote.close()


@requires_remote
@remote_group
def test_migration_007_fails_closed_on_legacy_rows():
    """Requirement 06 §12.6 #4/#5: anchor_id=0 rows and a non-empty legacy
    neighbor table without role_type abort the migration with the data
    untouched; once resolved, the migration applies cleanly."""
    schema = f"nz7b_rmt_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"
    remote = _Remote()
    try:
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            remote.run_file(_BASE_SQL_PATH, remap=schema)
            remote.query(f"DROP TABLE {schema}.neighbor_bitmaps, {schema}.cooccurrence_bitmaps")
            remote.query(
                f"CREATE TABLE {schema}.cooccurrence_bitmaps ("
                f"atom_id BIGINT, anchor_id BIGINT DEFAULT 0, "
                f"event_bitmap ROARINGBITMAP64 NOT NULL, PRIMARY KEY (atom_id, anchor_id))"
            )
            remote.query(
                f"CREATE TABLE {schema}.neighbor_bitmaps ("
                f"atom_id BIGINT, anchor_id BIGINT DEFAULT 0, "
                f"neighbor_bitmap ROARINGBITMAP64 NOT NULL, PRIMARY KEY (atom_id, anchor_id))"
            )
            atom_id = remote.query(
                f"INSERT INTO {schema}.atoms (text, atom_type) VALUES ('rmt07-legacy', 'E') RETURNING atom_id"
            )[0]
            remote.query(
                f"INSERT INTO {schema}.cooccurrence_bitmaps (atom_id, anchor_id, event_bitmap) "
                f"VALUES ({atom_id}, 0, rb64_build(ARRAY[1::bigint]))"
            )

            # anchor_id = 0 -> fail closed, data untouched.
            with pytest.raises(RuntimeError, match="anchor_id = 0"):
                remote.run_file(_MIGRATION_007_PATH, remap=schema)
            assert remote.query(f"SELECT count(*) FROM {schema}.cooccurrence_bitmaps WHERE anchor_id = 0") == ["1"]
            assert "role_type" not in remote.query(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_schema='{schema}' AND table_name='neighbor_bitmaps'"
            )

            # Non-empty legacy neighbor without role_type -> fail closed, no split.
            remote.query(f"DELETE FROM {schema}.cooccurrence_bitmaps")
            remote.query(
                f"INSERT INTO {schema}.neighbor_bitmaps (atom_id, anchor_id, neighbor_bitmap) "
                f"VALUES ({atom_id}, 5, rb64_build(ARRAY[2::bigint]))"
            )
            with pytest.raises(RuntimeError, match="role_type"):
                remote.run_file(_MIGRATION_007_PATH, remap=schema)
            assert remote.query(f"SELECT count(*) FROM {schema}.neighbor_bitmaps") == ["1"]

            # Resolved -> upgrade applies and is re-runnable.
            remote.query(f"DELETE FROM {schema}.neighbor_bitmaps")
            remote.run_file(_MIGRATION_007_PATH, remap=schema)
            remote.run_file(_MIGRATION_007_PATH, remap=schema)
            assert remote.query(
                f"SELECT is_nullable FROM information_schema.columns "
                f"WHERE table_schema='{schema}' AND table_name='cooccurrence_bitmaps' AND column_name='anchor_id'"
            ) == ["NO"]
        finally:
            remote.query(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        remote.close()


@requires_remote
@remote_group
def test_migration_006_catalog_guard_and_comment_idempotent():
    """Requirement 05A §8.2: migration 006 checks the shared-generation
    pre-state, aligns the embedding_profiles comment, and is safe to re-run
    without writing a single profile row."""
    schema = f"nz6_rmt_{random.choice(string.ascii_lowercase)}{uuid.uuid4().hex[:10]}"
    remote = _Remote()
    try:
        remote.query(f"CREATE SCHEMA {schema}")
        try:
            remote.run_file(_BASE_SQL_PATH, remap=schema)

            # 006 applies twice (idempotent) and leaves zero profile rows.
            remote.run_file(_MIGRATION_006_PATH, remap=schema)
            remote.run_file(_MIGRATION_006_PATH, remap=schema)
            assert remote.query(f"SELECT count(*) FROM {schema}.embedding_profiles") == ["0"]

            comment = remote.query(
                f"SELECT obj_description('{schema}.embedding_profiles'::regclass, 'pg_class')"
            )
            assert comment, "006 did not set the embedding_profiles comment"
            lowered = comment[0].lower()
            assert "shared embedding space" in lowered, f"006 comment lost the shared-space wording: {comment}"
            assert "no second context row" in lowered, f"006 comment must reject a second context row: {comment}"
            assert "not an identity-only scope" in lowered, f"006 comment must disclaim the identity-only scope: {comment}"
        finally:
            remote.query(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        remote.close()
