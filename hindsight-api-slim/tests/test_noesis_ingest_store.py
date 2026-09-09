"""Noesis store tests: mapping, envelope, sequence issue, failure injection.

Requirement 02 §10 / §17.4–§17.6 realigned to the 04A latest contract: events
are sequence-numbered (no ingestion_key, no replay, no collision — duplicate
input is a new event by contract), atoms carry no support_count, event_atoms
use role_type A/P/R/M plus target_occ. The database is the in-memory
FakeStore; the extraction layer is bypassed (components injected as golden
ExtractionOutcomes through the module-level extract seam).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from hyperextract.noesis import ExtractionAlert, FactComponent

import hindsight_api.engine.retain.noesis_ingest as noesis_ingest
from hindsight_api.engine.retain.noesis_ingest import ingest_noesis_batch
from tests.noesis_fakes import (
    CONTENT,
    GOLDEN_FACT_RECURSIVE_JSON,
    OBSERVED_AT,
    FakeExtractOnceFactory,
    FakeIdentityClient,
    FakeStore,
    golden_fact_recursive,
    golden_fact_time,
    identity_factory_for,
    llm_config,
    noesis_config,
    outcome,
    pool_factory_for,
)


async def run(store, contents, components=None, **kwargs):
    def fake_extract(text, *, extract_once):
        return outcome(list(components or []))

    noesis_ingest.extract_noesis_components = fake_extract
    # Default fake identity client: keeps these tests offline and alert-quiet;
    # requirement-03 vector behavior itself lives in test_noesis_identity_vector.py.
    kwargs.setdefault("identity_client_factory", identity_factory_for(FakeIdentityClient()))
    try:
        await ingest_noesis_batch(
            contents,
            "bank-1",
            noesis_config(),
            llm_config=llm_config(),
            extract_once_factory=FakeExtractOnceFactory(),
            pool_factory=pool_factory_for(store),
            **kwargs,
        )
    finally:
        noesis_ingest.extract_noesis_components = _ORIGINAL_EXTRACT


_ORIGINAL_EXTRACT = noesis_ingest.extract_noesis_components


def atom_upsert_args(store):
    """(text, atom_type) pairs of the atom upserts, in call order.

    The requirement-03 ``$3::vector`` third parameter is intentionally
    projected away here — its exact value is asserted by
    test_noesis_identity_vector.py; these requirement-02 assertions only pin
    which typed literals were upserted.
    """
    return [(args[0], args[1]) for kind, sql, args in store.calls if kind == "fetchrow" and ".atoms" in sql]


def event_atom_args(store):
    return [args for kind, sql, args in store.calls if kind == "execute" and ".event_atoms" in sql]


def event_insert_calls(store):
    return [
        (sql, args)
        for kind, sql, args in store.calls
        if kind == "fetchrow" and ".events" in sql and sql.lstrip().upper().startswith("INSERT")
    ]


def first_event_data(store) -> dict:
    return next(iter(store.events.values()))["data"]


def one_fact_contents(event_date=OBSERVED_AT):
    return [{"content": CONTENT, "event_date": event_date, "document_id": "doc-1"}]


# ---------------------------------------------------------------------------
# §10 database mapping
# ---------------------------------------------------------------------------

async def test_event_atoms_mechanical_mapping(monkeypatch):
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    rows = event_atom_args(store)
    assert len(rows) == 5
    by_occurrence = {row[1]: row for row in rows}
    # (event_id, occurrence_id, atom_id, role_type, target_occ)
    assert by_occurrence[1][3] == "A"  # agent
    assert by_occurrence[1][4] == 4
    assert by_occurrence[2][3] == "R"  # predicate
    assert by_occurrence[2][4] == 4
    assert by_occurrence[3][3] == "P"  # patient
    assert by_occurrence[3][4] == 2
    assert by_occurrence[4][3] == "R"
    assert by_occurrence[4][4] is None  # root predicate target is NULL
    assert by_occurrence[5][3] == "P"
    assert by_occurrence[5][4] == 4
    # anchor_id is never written (stays NULL): the insert carries five values only
    assert all(len(row) == 5 for row in rows)


async def test_role_four_values_map_one_to_one(monkeypatch):
    """agent→A, patient→P, predicate→R, modifier→M — every value exercised."""
    store = FakeStore()
    await run(
        store,
        [{"content": "昨天妈妈在超市买了苹果。", "event_date": OBSERVED_AT}],
        [golden_fact_time()],
    )

    rows = event_atom_args(store)
    by_occurrence = {row[1]: row for row in rows}
    assert by_occurrence[1][3] == "M"  # 昨天 modifier
    assert by_occurrence[2][3] == "A"  # 妈妈 agent
    assert by_occurrence[3][3] == "M"  # 在超市 modifier
    assert by_occurrence[4][3] == "R"  # 买 predicate
    assert by_occurrence[5][3] == "P"  # 苹果 patient


async def test_repeated_text_two_occurrences_one_upsert(monkeypatch):
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    assert len(event_atom_args(store)) == 5  # 小明 twice → two occurrences
    upserts = atom_upsert_args(store)
    # distinct typed literals: (小明,E), (没写,P), (作业,E), (揍,P) → 4 upserts
    assert set(upserts) == {("小明", "E"), ("作业", "E"), ("没写", "P"), ("揍", "P")}
    assert len(store.atoms) == 4  # one atom row per typed literal


async def test_same_text_different_types_are_distinct_atoms(monkeypatch):
    store = FakeStore()
    payload = {
        "utterance_type": "fact",
        "atoms": [
            {"pos": 1, "text": "每天", "type": "E", "role": "modifier", "target_occ": 2, "resolved": None},
            {"pos": 2, "text": "每天", "type": "P", "role": "predicate", "target_occ": None, "resolved": None},
        ],
        "tree": {
            "predicate": "每天",
            "agent": [],
            "patient": [],
            "modifier": ["每天"],
            "nested": [],
            "conditional": [],
        },
    }
    await run(store, [{"content": "每天，每天。", "event_date": OBSERVED_AT}], [FactComponent.model_validate(payload)])

    upserts = atom_upsert_args(store)
    assert sorted(upserts) == [("每天", "E"), ("每天", "P")]  # no cross-type merge
    assert store.atoms[("每天", "E")]["atom_id"] != store.atoms[("每天", "P")]["atom_id"]
    # two occurrences (same text, different types) → two event_atoms, two atoms
    assert len(event_atom_args(store)) == 2


async def test_g_type_atom_persisted_by_contract(monkeypatch):
    store = FakeStore()
    payload = json.loads(json.dumps(GOLDEN_FACT_RECURSIVE_JSON))
    payload["atoms"].append({"pos": 6, "text": "周末计划", "type": "G", "role": "modifier", "target_occ": 4, "resolved": None})
    await run(store, one_fact_contents(), [FactComponent.model_validate(payload)])

    assert ("周末计划", "G") in atom_upsert_args(store)


async def test_events_data_canonical_envelope(monkeypatch):
    store = FakeStore()
    await run(store, [{"content": CONTENT, "event_date": OBSERVED_AT, "document_id": "doc-9"}], [golden_fact_recursive()])

    data = first_event_data(store)
    assert data["contract_version"] == "noesis-event-closure-v1"
    assert data["bank_id"] == "bank-1"
    assert data["document_id"] == "doc-9"
    assert data["item_index"] == 0
    assert data["component_index"] == 0
    assert data["source_text"] == CONTENT
    assert data["observed_at"] == "2026-09-05T02:00:00Z"
    assert data["component"] == GOLDEN_FACT_RECURSIVE_JSON  # resolved/tree preserved verbatim
    assert data["time_resolution"]["strategy"] == "observed_at"


async def test_time_range_metadata_saved_in_data(monkeypatch):
    from tests.noesis_fakes import constraint

    class _DictAnalyzer:
        def __init__(self, mapping):
            self.mapping = mapping

        def analyze(self, query, reference_date=None):
            return self.mapping.get(query)

    store = FakeStore()
    analyzer = {"昨天": constraint(datetime(2026, 9, 4, 0, 0), datetime(2026, 9, 4, 23, 59, 59, 999999))}
    await run(
        store,
        [{"content": "昨天妈妈在超市买了苹果。", "event_date": OBSERVED_AT}],
        [golden_fact_time()],
        analyzer=_DictAnalyzer(analyzer),
    )
    data = first_event_data(store)
    resolution = data["time_resolution"]
    assert resolution["strategy"] == "modifier_atom"
    assert resolution["matched_atoms"] == ["昨天"]
    assert resolution["start"] == "2026-09-03T16:00:00Z"
    assert resolution["end"] == "2026-09-04T15:59:59.999999Z"
    assert resolution["fallback_reason"] is None
    event_time = next(iter(store.events.values()))["event_time"]
    assert event_time == datetime.fromisoformat("2026-09-03T16:00:00+00:00")


# ---------------------------------------------------------------------------
# 04A latest contract: sequence-numbered event insert
# ---------------------------------------------------------------------------

async def test_event_insert_sequence_returning_no_conflict(monkeypatch):
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    calls = event_insert_calls(store)
    assert len(calls) == 1
    sql, args = calls[0]
    assert "ingestion_key" not in sql
    assert "ON CONFLICT" not in sql
    assert "RETURNING event_id" in sql
    # (event_time, data_json, source, category) — source 'L' (LLM extraction),
    # category 'R' (raw) are the frozen "char" enum values of the new DDL.
    assert args[2] == "L"
    assert args[3] == "R"


async def test_event_id_shared_with_event_atoms_same_transaction(monkeypatch):
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    event_id = next(iter(store.events.values()))["event_id"]
    rows = event_atom_args(store)
    assert rows
    assert all(row[0] == event_id for row in rows)


async def test_atom_upsert_carries_no_support_count(monkeypatch):
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    upserts = [
        (sql, args)
        for kind, sql, args in store.calls
        if kind == "fetchrow" and ".atoms" in sql and sql.lstrip().upper().startswith("INSERT")
    ]
    assert upserts
    for sql, args in upserts:
        assert "support_count" not in sql
        assert len(args) == 3  # (text, atom_type, vector) only


# ---------------------------------------------------------------------------
# §11.2 duplicate input is a new event (04A contract behavior)
# ---------------------------------------------------------------------------

async def test_serial_duplicate_input_two_events(monkeypatch):
    """Events have no unique key: the same input twice lands two events."""
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    assert len(store.events) == 2
    assert len(store.event_atoms) == 10  # 5 per event
    assert len(store.atoms) == 4  # upserts are still per-typed-literal unique
    assert store.rollbacks == 0


async def test_duplicate_input_event_ids_differ(monkeypatch):
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    assert sorted(store.events) == sorted(set(store.events))  # distinct ids
    first_ids = {row[0] for row in event_atom_args(store)[:5]}
    second_ids = {row[0] for row in event_atom_args(store)[5:]}
    assert first_ids != second_ids  # each event's rows share only their own id


async def test_same_content_different_observed_at_two_events(monkeypatch):
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    await run(store, one_fact_contents(OBSERVED_AT + timedelta(days=1)), [golden_fact_recursive()])

    assert len(store.events) == 2
    assert len(store.atoms) == 4


async def test_concurrent_duplicate_input_two_events(monkeypatch):
    import asyncio

    async def yield_once():  # fresh coroutine per call: two tasks may interleave safely
        await asyncio.sleep(0)

    store = FakeStore()
    store.interleave = yield_once  # interleave the two coroutines at each statement
    await asyncio.gather(
        run(store, one_fact_contents(), [golden_fact_recursive()]),
        run(store, one_fact_contents(), [golden_fact_recursive()]),
    )

    assert len(store.events) == 2
    assert len(store.event_atoms) == 10
    assert len(store.atoms) == 4  # one atom row per typed literal


async def test_two_events_concurrent_same_atom_single_row(monkeypatch):
    import asyncio

    store = FakeStore()
    await asyncio.gather(
        run(store, one_fact_contents(), [golden_fact_recursive()]),
        run(store, one_fact_contents(OBSERVED_AT + timedelta(days=1)), [golden_fact_recursive()]),
    )
    assert len(store.events) == 2
    assert len(store.atoms) == 4  # one atom row per typed literal


async def test_extraction_error_retry_single_alert(monkeypatch):
    store = FakeStore()

    def failing_extract(text, *, extract_once):
        return outcome(
            alerts=[
                ExtractionAlert(
                    stage="hyper_extract",
                    alert_code="extraction_failed",
                    severity="error",
                    message="extraction failed after two attempts",
                    details={"error_type": "RuntimeError"},
                )
            ]
        )

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", failing_extract)
    for _ in range(2):
        await ingest_noesis_batch(
            one_fact_contents(),
            "bank-1",
            noesis_config(),
            llm_config=llm_config(),
            extract_once_factory=FakeExtractOnceFactory(),
            pool_factory=pool_factory_for(store),
            identity_client_factory=identity_factory_for(FakeIdentityClient()),
        )

    assert len(store.alerts_by_code("extraction_failed")) == 1  # deduped by key
    assert store.events == {}


# ---------------------------------------------------------------------------
# §11.3 / §17.6 failure injection — zero half state
# ---------------------------------------------------------------------------

def _assert_no_fact_state(store):
    assert store.events == {}, "event must roll back"
    assert store.atoms == {}, "atom rows must roll back"
    assert store.event_atoms == [], "event_atoms must roll back"


async def _run_with_failure(store, fail_after=None, fail_on_commit=False):
    store.fail_after = fail_after or {}
    store.fail_on_commit = fail_on_commit
    await run(store, one_fact_contents(), [golden_fact_recursive()])


async def test_failure_before_event_insert(monkeypatch):
    store = FakeStore()
    await _run_with_failure(store, fail_after={".events": 0})
    _assert_no_fact_state(store)
    assert len(store.alerts_by_code("event_ingest_failed")) == 1
    assert store.rollbacks >= 1


async def test_failure_after_event_insert_before_atoms(monkeypatch):
    store = FakeStore()
    await _run_with_failure(store, fail_after={".atoms": 0})
    _assert_no_fact_state(store)
    assert len(store.alerts_by_code("event_ingest_failed")) == 1


async def test_failure_after_atoms_before_event_atoms(monkeypatch):
    store = FakeStore()
    await _run_with_failure(store, fail_after={".event_atoms": 0})
    _assert_no_fact_state(store)
    assert len(store.alerts_by_code("event_ingest_failed")) == 1


async def test_failure_mid_event_atoms(monkeypatch):
    store = FakeStore()
    await _run_with_failure(store, fail_after={".event_atoms": 2})
    _assert_no_fact_state(store)
    assert len(store.alerts_by_code("event_ingest_failed")) == 1


async def test_failure_at_commit(monkeypatch):
    store = FakeStore()
    await _run_with_failure(store, fail_on_commit=True)
    _assert_no_fact_state(store)
    assert len(store.alerts_by_code("event_ingest_failed")) == 1


async def test_alert_write_failure_does_not_propagate(monkeypatch):
    store = FakeStore()
    store.fail_after = {".ingestion_alerts": 0}
    # independent alert writes fail — ingest must not raise
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    assert len(store.events) == 1  # the fact itself committed fine
    assert store.alerts == []
