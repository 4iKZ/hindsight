"""Noesis store tests: mapping, envelope, idempotency, failure injection.

Requirement 02 §10 / §11 / §17.4–§17.6. The database is the in-memory
FakeStore; the extraction layer is bypassed (components injected as golden
ExtractionOutcomes through the module-level extract seam).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

from hyperextract.noesis import ExtractionAlert, FactComponent

import hindsight_api.engine.retain.noesis_ingest as noesis_ingest
from hindsight_api.engine.retain.noesis_ingest import (
    NoesisInputItem,
    compute_ingestion_key,
    ingest_noesis_batch,
)
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
    # (event_id, occurrence_id, atom_id, role_type, head_occurrence_id)
    assert by_occurrence[1][3] == "agent"
    assert by_occurrence[1][4] == 4
    assert by_occurrence[2][3] == "predicate"
    assert by_occurrence[2][4] == 4
    assert by_occurrence[3][3] == "patient"
    assert by_occurrence[3][4] == 2
    assert by_occurrence[4][3] == "predicate"
    assert by_occurrence[4][4] is None  # root predicate head is NULL
    assert by_occurrence[5][3] == "patient"
    assert by_occurrence[5][4] == 4
    # anchor_id is never written (stays NULL): the insert carries five values only
    assert all(len(row) == 5 for row in rows)


async def test_repeated_text_two_occurrences_one_support(monkeypatch):
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    assert len(event_atom_args(store)) == 5  # 小明 twice → two occurrences
    upserts = atom_upsert_args(store)
    # distinct typed literals: (小明,E), (没写,P), (作业,E), (揍,P) → 4 upserts
    assert set(upserts) == {("小明", "E"), ("作业", "E"), ("没写", "P"), ("揍", "P")}
    assert store.support_count("小明", "E") == 1
    assert store.support_count("没写", "P") == 1


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
# §11.1 ingestion_key
# ---------------------------------------------------------------------------

def _item(**overrides):
    fields = dict(
        bank_id="bank-1",
        content=CONTENT,
        observed_at=OBSERVED_AT,
        operation_id=None,
        document_id=None,
        item_index=0,
    )
    fields.update(overrides)
    return NoesisInputItem(**fields)


def test_ingestion_key_stable_for_same_input():
    dump = golden_fact_recursive().model_dump(mode="json")
    assert compute_ingestion_key(item=_item(), component_index=0, component_json=dump) == compute_ingestion_key(
        item=_item(), component_index=0, component_json=dump
    )


def test_ingestion_key_varies_with_component_index():
    dump = golden_fact_recursive().model_dump(mode="json")
    assert compute_ingestion_key(item=_item(), component_index=0, component_json=dump) != compute_ingestion_key(
        item=_item(), component_index=1, component_json=dump
    )


def test_ingestion_key_varies_with_observed_at():
    dump = golden_fact_recursive().model_dump(mode="json")
    assert compute_ingestion_key(item=_item(), component_index=0, component_json=dump) != compute_ingestion_key(
        item=_item(observed_at=OBSERVED_AT + timedelta(minutes=1)), component_index=0, component_json=dump
    )


def test_ingestion_key_varies_with_content():
    dump = golden_fact_recursive().model_dump(mode="json")
    assert compute_ingestion_key(item=_item(), component_index=0, component_json=dump) != compute_ingestion_key(
        item=_item(content="另一句话。"), component_index=0, component_json=dump
    )


# ---------------------------------------------------------------------------
# §11.2/§11.4 idempotency & concurrency
# ---------------------------------------------------------------------------

async def test_serial_replay_single_event_no_double_support(monkeypatch):
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    assert len(store.events) == 1
    assert store.support_count("小明", "E") == 1
    assert len(store.event_atoms) == 5
    assert store.rollbacks == 0


async def test_replay_returns_existing_event_id(monkeypatch):
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    first_event_id = next(iter(store.events.values()))["event_id"]
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    assert next(iter(store.events.values()))["event_id"] == first_event_id


async def test_same_content_different_observed_at_two_events(monkeypatch):
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    await run(store, one_fact_contents(OBSERVED_AT + timedelta(days=1)), [golden_fact_recursive()])

    assert len(store.events) == 2
    assert store.support_count("小明", "E") == 2


async def test_concurrent_same_event_single_row(monkeypatch):
    async def yield_once():  # fresh coroutine per call: two tasks may interleave safely
        await asyncio.sleep(0)

    store = FakeStore()
    store.interleave = yield_once  # interleave the two coroutines at each statement
    await asyncio.gather(
        run(store, one_fact_contents(), [golden_fact_recursive()]),
        run(store, one_fact_contents(), [golden_fact_recursive()]),
    )

    assert len(store.events) == 1
    assert store.support_count("小明", "E") == 1
    assert len(store.event_atoms) == 5


async def test_two_events_concurrent_same_atom_support_two(monkeypatch):
    store = FakeStore()
    await asyncio.gather(
        run(store, one_fact_contents(), [golden_fact_recursive()]),
        run(store, one_fact_contents(OBSERVED_AT + timedelta(days=1)), [golden_fact_recursive()]),
    )
    assert len(store.events) == 2
    assert store.support_count("小明", "E") == 2
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


async def test_key_collision_fails_closed(monkeypatch):
    """Same ingestion key, different stored component → collision alert, old data intact."""
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    original_event_id = next(iter(store.events.values()))["event_id"]

    original_select = FakeStore._select_event

    def tampered_select(self, args):
        row = original_select(self, args)
        if row is not None:
            data = json.loads(row["data"])
            data["component"]["tree"]["predicate"] = "被篡改"
            row["data"] = json.dumps(data, ensure_ascii=False)
        return row

    monkeypatch.setattr(FakeStore, "_select_event", tampered_select)
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    collisions = store.alerts_by_code("ingestion_key_collision")
    assert len(collisions) == 1
    assert collisions[0]["severity"] == "error"
    assert collisions[0]["stage"] == "event_ingest"
    assert collisions[0]["event_id"] == original_event_id  # existing event recorded
    assert first_event_data(store)["component"]["tree"]["predicate"] == "揍"  # old data untouched


# ---------------------------------------------------------------------------
# §11.3 / §17.6 failure injection — zero half state
# ---------------------------------------------------------------------------

def _assert_no_fact_state(store):
    assert store.events == {}, "event must roll back"
    assert store.atoms == {}, "atom support must roll back"
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
