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
    FakeEmbeddingClient,
    FakeExtractOnceFactory,
    FakeStore,
    embedding_factory_for,
    golden_fact_pivot,
    golden_fact_recursive,
    golden_fact_time,
    llm_config,
    noesis_config,
    outcome,
    pool_factory_for,
)


async def run(store, contents, components=None, **kwargs):
    def fake_extract(text, *, extract_once):
        return outcome(list(components or []))

    noesis_ingest.extract_noesis_components = fake_extract
    # Default fake embedding client: keeps these tests offline and alert-quiet;
    # requirement-03 vector behavior itself lives in test_noesis_embedding.py.
    kwargs.setdefault("embedding_client_factory", embedding_factory_for(FakeEmbeddingClient()))
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
    test_noesis_embedding.py; these requirement-02 assertions only pin
    which typed literals were upserted. The requirement-05 atom row lock
    (``SELECT ... FROM {s}.atoms ... FOR UPDATE``) also matches ``".atoms"``
    but is not an upsert, so it is excluded here.
    """
    return [
        (args[0], args[1])
        for kind, sql, args in store.calls
        if kind == "fetchrow" and ".atoms" in sql and "FOR UPDATE" not in sql
    ]


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
    # Requirement 05 contract: the insert carries six values — anchor_id (row[5])
    # is always routed to a concrete anchor, never left NULL.
    assert all(len(row) == 6 for row in rows)
    assert all(row[5] is not None for row in rows)


async def test_event_atoms_anchor_id_is_routed_non_null(monkeypatch):
    """Requirement 05: every event_atom insert carries a routed anchor_id."""
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    rows = event_atom_args(store)
    assert rows
    for row in rows:
        assert len(row) == 6  # (event_id, occurrence_id, atom_id, role_type, target_occ, anchor_id)
        assert row[5] is not None


async def test_shared_pivot_borrows_real_anchor_without_a_second_row(monkeypatch):
    """需求 08 §12.4: one row per real atom, the pivot reuses its physical anchor,
    and the borrowed member adds neither an event_atom row nor a bitmap key."""
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_pivot()])

    rows = event_atom_args(store)
    assert len(rows) == 5  # 妈妈 让 小明 打 酱油 — no implied duplicate row
    by_occurrence = {row[1]: row for row in rows}
    ming_id = by_occurrence[3][2]
    assert by_occurrence[4][4] == 3  # 打 targets 小明, not the parent predicate
    assert sum(1 for row in rows if row[2] == ming_id) == 1
    assert all(isinstance(row[5], int) and row[5] > 0 for row in rows)

    ming_anchor_rows = [row for row in store.anchors_rows() if row["atom_id"] == ming_id]
    assert len(ming_anchor_rows) == 1
    assert ming_anchor_rows[0]["total_count"] == 1  # routed once with the parent frame
    ming_anchor = ming_anchor_rows[0]["anchor_id"]
    assert by_occurrence[3][5] == ming_anchor

    cooccurrence_keys = {(atom_id, anchor_id) for atom_id, anchor_id in store.cooccurrence_bitmaps}
    assert len(cooccurrence_keys) == 5
    assert len({key for key in cooccurrence_keys if key[0] == ming_id}) == 1

    ming_neighbors = store.neighbor_bitmaps.get((ming_id, ming_anchor, "N"))
    assert ming_neighbors == {by_occurrence[1][2], by_occurrence[2][2], by_occurrence[4][2], by_occurrence[5][2]}
    beat_s = store.neighbor_bitmaps.get((by_occurrence[4][2], by_occurrence[4][5], "S"))
    assert beat_s == {ming_id}


async def test_event_insert_carries_no_context_vector(monkeypatch):
    """events.context_embedding is not written by this chain (requirement 05 §8.4)."""
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    calls = event_insert_calls(store)
    assert calls
    for sql, args in calls:
        assert "context_embedding" not in sql
        assert len(args) == 4  # (event_time, data_json, source, category) only


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


async def test_atom_upserts_use_deterministic_literal_order(monkeypatch):
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    upserts = atom_upsert_args(store)
    assert upserts == sorted(upserts)


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


async def test_g_type_atom_aborts_component_with_frame_alert(monkeypatch):
    """Requirement 05 §2.3/§5.1: an external-G atom in a fact component means
    the upstream closure contract broke — the whole component is dropped with
    an anchor_frame_invalid alert instead of persisting an unroutable atom."""
    store = FakeStore()
    payload = json.loads(json.dumps(GOLDEN_FACT_RECURSIVE_JSON))
    payload["atoms"].append({"pos": 6, "text": "周末计划", "type": "G", "role": "modifier", "target_occ": 4, "resolved": None})
    await run(store, one_fact_contents(), [FactComponent.model_validate(payload)])

    assert atom_upsert_args(store) == []  # nothing persists
    assert store.events == {}
    assert store.event_atoms == []
    alerts = store.alerts_by_code("anchor_frame_invalid")
    assert len(alerts) == 1
    assert alerts[0]["stage"] == "anchor_context"
    assert alerts[0]["severity"] == "error"
    assert alerts[0]["details"]["component_index"] == 0
    assert alerts[0]["details"]["predicate_pos"] == 6
    assert alerts[0]["details"]["exception_type"] == "AnchorFrameError"


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
            embedding_client_factory=embedding_factory_for(FakeEmbeddingClient()),
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
    assert store.anchors == {}, "anchors must roll back (requirement 05 §8.3)"


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


# ---------------------------------------------------------------------------
# Requirement 05 §13.5: anchor routing on the ingest main chain
# ---------------------------------------------------------------------------

async def test_event_atom_anchor_ids_match_routed_anchors(monkeypatch):
    """§13.5 #2: golden_fact_time's five E/P atoms all live in frame 4 → five
    anchors each with total_count=1, and every event_atom row's anchor_id is
    the anchor routed for exactly its (atom_id, frame)."""
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_time()])

    anchors = store.anchors_rows()
    assert len(anchors) == 5  # five distinct typed literals, one route each
    assert all(anchor["total_count"] == 1 for anchor in anchors)
    rows = event_atom_args(store)
    assert len(rows) == 5
    anchor_by_id = {anchor["anchor_id"]: anchor for anchor in anchors}
    for row in rows:
        assert len(row) == 6
        anchor = anchor_by_id[row[5]]  # row[5] is the routed anchor_id
        assert anchor["atom_id"] == row[2]  # routed for this very atom row


async def test_context_embed_calls_precede_every_fact_transaction_begin(monkeypatch):
    """§13.5 #4 (05A §5.1 realignment): /normalize/sentence HTTP never runs
    inside the FACT transaction. The 05A frozen order moved the shared
    profile gate (and its empty-store claim transaction) BEFORE the Context
    calls, so an earlier claim begin is legal — what stays frozen is that
    every context embed pre-dates the fact transaction's begin."""
    store = FakeStore()
    client = FakeEmbeddingClient()
    original_embed = client.embed_context

    async def recording_embed(context_text):
        store.tx_log.append(("context_embed", context_text))
        return await original_embed(context_text)

    client.embed_context = recording_embed
    await run(store, one_fact_contents(), [golden_fact_recursive()], embedding_client_factory=embedding_factory_for(client))

    embed_positions = [index for index, entry in enumerate(store.tx_log) if isinstance(entry, tuple)]
    assert len(embed_positions) == 2  # golden_fact_recursive has two frames
    begins = [index for index, entry in enumerate(store.tx_log) if entry == "begin"]
    assert len(begins) == 2  # one claim transaction + one fact transaction
    fact_begin = begins[-1]
    assert max(embed_positions) < fact_begin  # all context HTTP pre-dates the fact begin


async def test_anchor_sql_runs_only_inside_transaction(monkeypatch):
    """§13.5 #5: every anchors read/write and every FOR UPDATE atom lock is
    dispatched while the fact transaction is open (_tx_depth > 0)."""
    store = FakeStore()
    anchor_sql_depths: list[int] = []
    original_fetchrow = store.fetchrow
    original_execute = store.execute

    async def spy_fetchrow(sql, *args):
        if ".anchors" in sql or "FOR UPDATE" in sql:
            anchor_sql_depths.append(store._tx_depth)
        return await original_fetchrow(sql, *args)

    async def spy_execute(sql, *args):
        if ".anchors" in sql or "FOR UPDATE" in sql:
            anchor_sql_depths.append(store._tx_depth)
        return await original_execute(sql, *args)

    store.fetchrow = spy_fetchrow
    store.execute = spy_execute
    await run(store, one_fact_contents(), [golden_fact_time()])

    # five (atom, frame) routes × (lock + nearest + insert), no EMA on fresh anchors
    assert len(anchor_sql_depths) == 15
    assert all(depth > 0 for depth in anchor_sql_depths)


async def test_anchor_insert_failure_rolls_back_whole_fact(monkeypatch):
    """§13.5 #8: an anchors INSERT failure rolls back the event, atoms,
    anchors, and event_atoms of this component with an anchor_route_failed
    alert (§9.2)."""
    store = FakeStore()
    await _run_with_failure(store, fail_after={"INSERT INTO noesis_core.anchors": 0})
    _assert_no_fact_state(store)
    alerts = store.alerts_by_code("anchor_route_failed")
    assert len(alerts) == 1
    assert alerts[0]["stage"] == "anchor_route"
    assert alerts[0]["severity"] == "error"
    assert alerts[0]["details"]["component_index"] == 0


async def test_ema_update_failure_rolls_back_whole_fact(monkeypatch):
    """§13.5 #9: an EMA centroid UPDATE failure (reuse path) rolls back the
    whole second component while the first component's committed anchors stay
    untouched (total_count unchanged, no half-EMA)."""
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    committed_anchors = {anchor["anchor_id"]: anchor["total_count"] for anchor in store.anchors_rows()}
    assert committed_anchors and all(count == 1 for count in committed_anchors.values())

    # Same input again: identical context vectors → distance 0 → reuse + EMA,
    # and the EMA UPDATE is the failing statement.
    store.fail_after = {"total_count = total_count + 1": 0}
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    assert len(store.events) == 1  # the second event rolled back
    assert len(store.event_atoms) == 5
    assert {anchor["anchor_id"]: anchor["total_count"] for anchor in store.anchors_rows()} == committed_anchors
    assert len(store.alerts_by_code("anchor_route_failed")) == 1


# ---------------------------------------------------------------------------
# Requirement 05A §11.2: in-transaction unconditional generation fence
# ---------------------------------------------------------------------------

_GOLDEN_LITERALS = [("小明", "E"), ("没写", "P"), ("作业", "E"), ("揍", "P")]


def _ready_profile(model="bge-m3", revision="bge-m3-1024-v1", dimension=1024):
    return {
        "embedding_kind": "identity",
        "model_name": model,
        "model_revision": revision,
        "dimension": dimension,
        "status": "ready",
    }


def _for_share_calls(store):
    return [call for call in store.calls if "FOR SHARE" in call[1] and ".embedding_profiles" in call[1]]


def _fence_alert(store):
    alerts = store.alerts_by_code("embedding_profile_unavailable")
    assert len(alerts) == 1
    return alerts[0]


# §11.2 #1
async def test_fence_for_share_runs_when_all_atoms_already_embedded(monkeypatch):
    """The fence is unconditional: even when every atom already carries its
    vector and this round makes zero Identity HTTP calls, the fact
    transaction still takes the profile FOR SHARE lock (05A §5.5)."""
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    fence_calls_before = len(_for_share_calls(store))

    second = FakeEmbeddingClient()
    await run(store, one_fact_contents(), [golden_fact_recursive()], embedding_client_factory=embedding_factory_for(second))

    assert second.identity_calls == []  # precheck saw has_embedding → zero bge
    assert len(_for_share_calls(store)) == fence_calls_before + 1  # fence still ran
    assert len(store.events) == 2  # both facts committed


# §11.2 #2
async def test_fence_for_share_runs_when_identity_prep_empty(monkeypatch):
    """Identity preparation degrades to empty (whole batch transport-failed);
    the fence still runs and the fact still lands (§5.4 keeps the identity
    NULL degradation — the fence is not conditioned on it)."""
    store = FakeStore()
    failing = FakeEmbeddingClient(identity_failures={literal: "timeout" for literal in _GOLDEN_LITERALS})
    await run(store, one_fact_contents(), [golden_fact_recursive()], embedding_client_factory=embedding_factory_for(failing))

    assert len(store.events) == 1  # identity failure degrades, fact commits
    assert len(_for_share_calls(store)) == 1  # fence ran despite empty vectors
    assert len(store.alerts_by_code("identity_vector_unavailable")) == 1


# §11.2 #3
async def test_fence_for_share_precedes_event_insert(monkeypatch):
    """05A §5.5: the FOR SHARE fence is the FIRST business statement of the
    fact transaction, before the event INSERT."""
    store = FakeStore()
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    fence_index = next(
        index for index, call in enumerate(store.calls) if call in _for_share_calls(store)
    )
    event_index = next(
        index
        for index, call in enumerate(store.calls)
        if call[0] == "fetchrow" and ".events" in call[1] and call[1].lstrip().upper().startswith("INSERT")
    )
    assert fence_index < event_index


# §11.2 #4
async def test_fence_profile_flipped_to_rebuilding_rolls_back_whole_fact(monkeypatch):
    """A rebuild that flips the profile to rebuilding after the outer gate
    but before the fence is caught in-transaction: the whole component rolls
    back with zero fact state and the fixed alert (05A §5.5/§5.6)."""
    store = FakeStore()
    store.profile = _ready_profile()

    async def flip_to_rebuilding():
        store.profile["status"] = "rebuilding"

    store.interleave = flip_to_rebuilding
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    _assert_no_fact_state(store)
    assert store.rollbacks >= 1
    alert = _fence_alert(store)
    assert alert["details"]["reason"] == "rebuilding"
    assert alert["details"]["db_profile"]["status"] == "rebuilding"


# §11.2 #5
async def test_fence_profile_switched_generation_rolls_back_whole_fact(monkeypatch):
    """The profile switches to another generation between the outer gate and
    the fence: the in-transaction re-check refuses, zero facts persist."""
    store = FakeStore()
    store.profile = _ready_profile()

    async def switch_generation():
        store.profile["model_name"] = "switched-model"

    store.interleave = switch_generation
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    _assert_no_fact_state(store)
    assert store.rollbacks >= 1
    alert = _fence_alert(store)
    assert alert["details"]["reason"] == "generation_mismatch"
    assert alert["details"]["db_profile"]["model_name"] == "switched-model"


# §11.2 #6
async def test_rebuild_status_update_waits_for_in_flight_share_lock(monkeypatch):
    """Fake-layer ordering proof of 05A §5.6: while the fact transaction holds
    the profile FOR SHARE lock, the rebuild ``SET status='rebuilding'``
    UPDATE cannot run — it is only issued after COMMIT. (The fake executes
    sequentially; the real row-lock mutual exclusion is validated against
    remote PostgreSQL by Task 8.)"""
    store = FakeStore()
    store.profile = _ready_profile()
    order: list[str] = []

    async def note_share_lock_held():
        # Runs before the first in-transaction statement: the FOR SHARE fence
        # below acquires the share lock, so a concurrent rebuild UPDATE must
        # be deferred until the transaction commits.
        if "share_lock_held" not in order:
            order.append("share_lock_held")

    store.interleave = note_share_lock_held
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    assert order == ["share_lock_held"]
    assert store.tx_log[-1] == "commit"  # share lock released at commit
    assert len(store.events) == 1

    # Only after the fact transaction committed can the rebuild UPDATE run.
    await store.execute(
        "UPDATE noesis_core.embedding_profiles SET status = 'rebuilding' WHERE embedding_kind = 'identity'"
    )
    order.append("rebuild_update_applied")
    assert order == ["share_lock_held", "rebuild_update_applied"]
    assert store.profile["status"] == "rebuilding"


# §11.2 #7
async def test_rebuild_update_succeeds_after_share_lock_release(monkeypatch):
    """Once the share lock is released the rebuild UPDATE succeeds, and every
    new fact component is then rejected by the outer gate before any bge
    call (05A §6.5)."""
    store = FakeStore()
    store.profile = _ready_profile()
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    assert len(store.events) == 1

    await store.execute(
        "UPDATE noesis_core.embedding_profiles SET status = 'rebuilding' WHERE embedding_kind = 'identity'"
    )
    client = FakeEmbeddingClient()
    await run(store, one_fact_contents(), [golden_fact_recursive()], embedding_client_factory=embedding_factory_for(client))

    assert client.context_calls == [] and client.identity_calls == []  # gate first, zero bge
    assert len(store.events) == 1  # no new facts during the rebuild
    alert = _fence_alert(store)
    assert alert["details"]["reason"] == "rebuilding"


# §11.2 #8
async def test_fence_rollback_leaves_anchors_untouched_no_ema_interleave(monkeypatch):
    """The fence raises before the event insert, so no Anchor EMA or create
    can interleave with the rebuild cutover: committed anchors are bit-for-bit
    unchanged after the fenced component rolls back."""
    store = FakeStore()
    store.profile = _ready_profile()
    await run(store, one_fact_contents(), [golden_fact_recursive()])
    committed = {
        anchor["anchor_id"]: (anchor["centroid"], anchor["total_count"]) for anchor in store.anchors_rows()
    }
    assert committed and all(count == 1 for _, count in committed.values())

    async def flip_to_rebuilding():
        store.profile["status"] = "rebuilding"

    store.interleave = flip_to_rebuilding
    await run(store, one_fact_contents(), [golden_fact_recursive()])

    assert len(store.events) == 1  # only the first, committed fact
    assert store.rollbacks >= 1
    assert {
        anchor["anchor_id"]: (anchor["centroid"], anchor["total_count"]) for anchor in store.anchors_rows()
    } == committed  # no EMA, no new anchors — zero cutover interleave
