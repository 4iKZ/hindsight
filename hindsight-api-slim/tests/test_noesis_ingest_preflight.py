"""R02-06 read-only schema preflight unit tests.

Uses a scriptable catalog-fake connection so each "missing object" branch is
verified precisely without touching the remote DB. The preflight function is
read-only by construction (SELECT on information_schema / pg_catalog only), so
these tests also assert that no DDL is ever issued.
"""

from __future__ import annotations

import pytest

import hindsight_api.engine.retain.noesis_ingest as noesis_ingest
from hindsight_api.engine.retain.noesis_ingest import SchemaPreflightError, _ensure_schema_ready, _run_schema_preflight


class _CatalogConn:
    """Implements fetchval for the preflight's SELECT catalog queries.

    ``missing`` is a set of object hints (schema/table/column/constraint names)
    whose existence should report False; everything else reports True.
    """

    def __init__(self, missing) -> None:
        self.missing = set(missing)
        self.sqls: list[str] = []

    async def fetchval(self, sql, *args):
        self.sqls.append(sql)
        if "format_type" in sql:
            # Requirement 03 §11: the declared atoms.embedding width. The fake
            # catalog models the fully migrated VECTOR(1024) column; tests can
            # override with ``missing={"atoms_embedding_dimension"}``.
            if "centroid_vector" in sql:
                # Requirement 05 §10.3: the declared anchors.centroid_vector
                # width (override: ``missing={"centroid_vector_dimension"}``).
                return "vector(768)" if "centroid_vector_dimension" in self.missing else "vector(1024)"
            return "vector(768)" if "atoms_embedding_dimension" in self.missing else "vector(1024)"
        if "pg_extension" in sql:
            return args[0] not in self.missing
        if "timescaledb_information.hypertables" in sql:
            return "events_hypertable" not in self.missing
        if "timescaledb_information.jobs" in sql:
            return "events_retention_90_days" not in self.missing
        if "pg_indexes" in sql:
            # 04A: the plain event_id index; requirement 05: the anchors
            # atom_id-leading index.
            if "idx_anchors_atom" in sql:
                return "idx_anchors_atom" not in self.missing
            return "idx_events_event_id" not in self.missing
        if "information_schema.schemata" in sql:
            return args[0] not in self.missing
        if "information_schema.tables" in sql:
            # args = (schema, table)
            return args[1] not in self.missing
        if "information_schema.columns" in sql and "is_nullable" in sql:
            # nullable probes name their column literally in the SQL
            for column, hint in (
                ("centroid_vector", "centroid_vector_nullable"),
                ("total_count", "total_count_nullable"),
                ("anchor_id", "anchor_id_nullable"),
                ("dedupe_key", "dedupe_key_nullable"),
            ):
                if f"'{column}'" in sql:
                    return hint not in self.missing
            return "dedupe_key_nullable" not in self.missing
        if "information_schema.columns" in sql:
            # existence probe: args = (schema, table, column)
            return args[2] not in self.missing
        if "pg_constraint" in sql and "contype = 'f'" in sql:
            # FK probes (requirement 05): args = (referencing, referenced)
            pair = tuple(part.rsplit(".", 1)[-1] for part in args[:2])
            if pair == ("anchors", "atoms"):
                return "anchors_atom_fk" not in self.missing
            if pair == ("event_atoms", "anchors"):
                return "event_atoms_anchor_fk" not in self.missing
            return True
        if "pg_constraint" in sql and "to_regclass" in sql:
            if "ingestion_alerts_dedupe_key_unique" in sql:
                return "ingestion_alerts_dedupe_key_unique" not in self.missing
            if "atoms_text_type_unique" in sql:
                return "atoms_text_type_unique" not in self.missing
        return True


class _CatalogPool:
    def __init__(self, missing) -> None:
        self._conn = _CatalogConn(missing)

    def acquire(self):
        return _AcquireCtx(self._conn)


class _AcquireCtx:
    def __init__(self, conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc) -> bool:
        return False


def _pool(missing=()):
    return _CatalogPool(missing)


@pytest.fixture(autouse=True)
def _reset_preflight_state():
    noesis_ingest._preflight_cache.clear()
    yield
    noesis_ingest._preflight_cache.clear()


async def _run(missing):
    await _run_schema_preflight(_pool(missing), "noesis_core")


# ---------------------------------------------------------------------------
# Individual missing-object branches
# ---------------------------------------------------------------------------

async def test_complete_schema_passes():
    await _run([])  # no exception


async def test_missing_schema_fails():
    with pytest.raises(SchemaPreflightError, match="schema 'noesis_core' does not exist"):
        await _run(["noesis_core"])


async def test_missing_required_extension_fails():
    with pytest.raises(SchemaPreflightError, match="extension 'roaringbitmap'"):
        await _run(["roaringbitmap"])


async def test_missing_events_hypertable_fails():
    with pytest.raises(SchemaPreflightError, match="TimescaleDB hypertable"):
        await _run(["events_hypertable"])


async def test_missing_90_day_retention_fails():
    with pytest.raises(SchemaPreflightError, match="90-day retention"):
        await _run(["events_retention_90_days"])


async def test_missing_table_fails():
    with pytest.raises(SchemaPreflightError, match="table 'noesis_core.atoms' does not exist"):
        await _run(["atoms"])


async def test_missing_dedupe_key_column_fails():
    with pytest.raises(SchemaPreflightError, match="column 'noesis_core.ingestion_alerts.dedupe_key' does not exist"):
        await _run(["dedupe_key"])


async def test_missing_events_event_id_index_fails():
    """04A: events carries no unique constraint; the plain event_id index is
    the required object the preflight checks instead."""
    with pytest.raises(SchemaPreflightError, match="idx_events_event_id"):
        await _run(["idx_events_event_id"])


async def test_ingestion_key_column_not_required():
    """04A: the dropped events.ingestion_key column must not be probed."""
    await _run(["ingestion_key"])  # no exception


async def test_head_occurrence_id_column_not_required():
    """04A: event_atoms uses target_occ; the old head column is not probed."""
    await _run(["head_occurrence_id"])  # no exception


async def test_missing_event_atoms_target_occ_fails():
    with pytest.raises(SchemaPreflightError, match="column 'noesis_core.event_atoms.target_occ' does not exist"):
        await _run(["target_occ"])


# ---------------------------------------------------------------------------
# Requirement 05 §10.3: anchor routing objects
# ---------------------------------------------------------------------------

async def test_missing_anchors_table_fails():
    with pytest.raises(SchemaPreflightError, match="table 'noesis_core.anchors' does not exist"):
        await _run(["anchors"])


async def test_missing_anchors_centroid_vector_column_fails():
    with pytest.raises(SchemaPreflightError, match="column 'noesis_core.anchors.centroid_vector' does not exist"):
        await _run(["centroid_vector"])


async def test_missing_anchor_id_column_fails():
    with pytest.raises(SchemaPreflightError, match="column 'noesis_core.event_atoms.anchor_id' does not exist"):
        await _run(["anchor_id"])


async def test_missing_context_embedding_column_fails():
    """Requirement 05: events.context_embedding must exist (existence only —
    the column legitimately stays NULL in Stage 1)."""
    with pytest.raises(SchemaPreflightError, match="column 'noesis_core.events.context_embedding' does not exist"):
        await _run(["context_embedding"])


async def test_wrong_centroid_dimension_fails():
    with pytest.raises(
        SchemaPreflightError, match=r"'noesis_core\.anchors\.centroid_vector' declares 'vector\(768\)'"
    ):
        await _run(["centroid_vector_dimension"])


async def test_centroid_vector_nullable_fails():
    with pytest.raises(SchemaPreflightError, match=r"'noesis_core\.anchors\.centroid_vector' is nullable"):
        await _run(["centroid_vector_nullable"])


async def test_total_count_nullable_fails():
    with pytest.raises(SchemaPreflightError, match=r"'noesis_core\.anchors\.total_count' is nullable"):
        await _run(["total_count_nullable"])


async def test_missing_anchors_atom_fk_fails():
    with pytest.raises(
        SchemaPreflightError, match="foreign key from 'noesis_core.anchors' to 'noesis_core.atoms' is missing"
    ):
        await _run(["anchors_atom_fk"])


async def test_anchor_id_nullable_fails():
    with pytest.raises(SchemaPreflightError, match=r"'noesis_core\.event_atoms\.anchor_id' is nullable"):
        await _run(["anchor_id_nullable"])


async def test_missing_event_atoms_anchor_fk_fails():
    with pytest.raises(
        SchemaPreflightError, match="foreign key from 'noesis_core.event_atoms' to 'noesis_core.anchors' is missing"
    ):
        await _run(["event_atoms_anchor_fk"])


async def test_missing_anchors_atom_index_fails():
    with pytest.raises(SchemaPreflightError, match="idx_anchors_atom"):
        await _run(["idx_anchors_atom"])


async def test_missing_atom_unique_constraint_fails():
    with pytest.raises(SchemaPreflightError, match="atoms_text_type_unique"):
        await _run(["atoms_text_type_unique"])


async def test_dedupe_key_nullable_fails():
    with pytest.raises(SchemaPreflightError, match="dedupe_key is not NOT NULL"):
        await _run(["dedupe_key_nullable"])


# ---------------------------------------------------------------------------
# Ensure-ready semantics + read-only guarantee
# ---------------------------------------------------------------------------

async def test_ensure_ready_fail_closes_on_named_missing_object():
    assert await _ensure_schema_ready(_pool(["dedupe_key"]), "noesis_core") is False


async def test_ensure_ready_transient_error_skips_noesis_and_retries():
    class _BoomPool:
        def acquire(self):
            raise RuntimeError("connector down")

    assert await _ensure_schema_ready(_BoomPool(), "noesis_core") is False
    assert await _ensure_schema_ready(_pool(), "noesis_core") is True


async def test_preflight_cache_is_scoped_to_pool_and_schema():
    first = _pool()
    second = _pool(["atoms"])
    assert await _ensure_schema_ready(first, "noesis_core") is True
    assert await _ensure_schema_ready(second, "noesis_core") is False


async def test_preflight_is_read_only():
    """The preflight only issues SELECT catalog queries; never any DDL."""
    pool = _pool([])
    await _run_schema_preflight(pool, "noesis_core")
    for sql in pool._conn.sqls:
        assert sql.lstrip().upper().startswith("SELECT"), f"preflight issued DDL: {sql}"
