"""Requirement 06 bitmap write-maintenance tests (planner / writer / transaction).

Pure planner units (requirement 06 §5/§12.1), nested-frame isolation (§12.2),
the database-side upsert writer against the in-memory FakeStore (§12.3),
transaction failure injection through the real ``ingest_noesis_batch`` chain
(§12.4), and concurrent union behavior (§12.5). No network, no real database.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from hyperextract.noesis import FactComponent

import hindsight_api.engine.retain.noesis_ingest as noesis_ingest
from hindsight_api.engine.retain.noesis_anchor import (
    AnchorPlan,
    OccurrenceRoute,
    PredicateFrame,
    SemanticFrameMember,
    plan_anchor_routing,
)
from hindsight_api.engine.retain.noesis_bitmap import (
    BitmapPlanError,
    BitmapWriteError,
    BitmapWritePlan,
    CooccurrenceBitmapWrite,
    NeighborBitmapWrite,
    apply_bitmap_writes,
    plan_bitmap_writes,
)
from hindsight_api.engine.retain.noesis_ingest import ingest_noesis_batch
from tests.noesis_fakes import (
    CONTENT,
    OBSERVED_AT,
    FakeConn,
    FakeEmbeddingClient,
    FakeExtractOnceFactory,
    FakeStore,
    embedding_factory_for,
    golden_fact_recursive,
    golden_fact_time,
    llm_config,
    noesis_config,
    outcome,
    pool_factory_for,
)

SCHEMA = "noesis_core"

_ORIGINAL_EXTRACT = noesis_ingest.extract_noesis_components


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def atom(pos, text, type_, role, target):
    return {"pos": pos, "text": text, "type": type_, "role": role, "target_occ": target, "resolved": None}


def fact(atoms):
    """FactComponent from atom dicts; the tree mirrors the root frame only."""
    root = next(a for a in atoms if a["role"] == "predicate" and a["target_occ"] is None)
    direct = [a for a in atoms if a["target_occ"] == root["pos"]]
    return FactComponent.model_validate(
        {
            "utterance_type": "fact",
            "atoms": atoms,
            "tree": {
                "predicate": root["text"],
                "agent": [{"text": a["text"], "modifier": [], "implied": False} for a in direct if a["role"] == "agent"],
                "patient": [
                    {"text": a["text"], "modifier": [], "implied": False} for a in direct if a["role"] == "patient"
                ],
                "modifier": [a["text"] for a in direct if a["role"] == "modifier"],
                "nested": [],
                "conditional": [],
            },
        }
    )


def planned(component, *, event_id: int = 77):
    """Run the real frame planner, then assign deterministic IDs/routes.

    Every occurrence literal gets a stable atom_id (100 + first-appearance
    index); every predicate frame gets its own anchor (900 + frame_pos), so
    cross-frame cases always route to distinct anchors.
    """
    anchor_plan = plan_anchor_routing(component.atoms)
    atom_ids: dict[tuple[str, str], int] = {}
    for occurrence in anchor_plan.occurrences:
        atom_ids.setdefault(occurrence.literal, 100 + len(atom_ids))
    anchor_routes: dict[tuple[int, int], int] = {}
    for occurrence in anchor_plan.occurrences:
        anchor_routes[(atom_ids[occurrence.literal], occurrence.frame_pos)] = 900 + occurrence.frame_pos
    plan = plan_bitmap_writes(
        event_id=event_id,
        atoms=component.atoms,
        anchor_plan=anchor_plan,
        atom_ids=atom_ids,
        anchor_routes=anchor_routes,
    )
    return plan, atom_ids, anchor_routes


def neighbor_rows(plan) -> dict[tuple[int, str], tuple[int, ...]]:
    return {(write.atom_id, write.role_type): write.neighbor_atom_ids for write in plan.neighbors}


async def run_ingest(store, monkeypatch, components=None, **kwargs):
    def fake_extract(text, *, extract_once):
        return outcome(list(components or []))

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", fake_extract)
    return await ingest_noesis_batch(
        [{"content": CONTENT, "event_date": OBSERVED_AT, "document_id": "doc-1"}],
        "bank-1",
        noesis_config(),
        llm_config=llm_config(),
        extract_once_factory=FakeExtractOnceFactory(),
        pool_factory=pool_factory_for(store),
        embedding_client_factory=kwargs.pop(
            "embedding_client_factory", embedding_factory_for(FakeEmbeddingClient())
        ),
        **kwargs,
    )


def bitmap_calls(store):
    return [
        (sql, args)
        for kind, sql, args in store.calls
        if kind == "execute" and (".cooccurrence_bitmaps" in sql or ".neighbor_bitmaps" in sql)
    ]


def _assert_no_fact_state(store):
    assert store.events == {}, "event must roll back"
    assert store.atoms == {}, "atom rows must roll back"
    assert store.anchors == {}, "anchors must roll back"
    assert store.event_atoms == [], "event_atoms must roll back"
    assert store.cooccurrence_bitmaps == {}, "cooccurrence bitmaps must roll back"
    assert store.neighbor_bitmaps == {}, "neighbor bitmaps must roll back"


# ---------------------------------------------------------------------------
# §12.1 pure planning units
# ---------------------------------------------------------------------------


def test_cooccurrence_writes_every_ep_occurrence_with_frame_anchor():
    plan, ids, _ = planned(golden_fact_time(), event_id=881)

    keys = [(write.atom_id, write.anchor_id) for write in plan.cooccurrences]
    assert keys == sorted(keys), "cooccurrence rows must be (atom_id, anchor_id) ascending"
    assert keys == sorted((ids[literal], 904) for literal in ids), "all five E/P occurrences incl. modifiers"
    assert {write.event_id for write in plan.cooccurrences} == {881}


def test_neighbor_e_rows_are_bidirectional_n():
    plan, ids, _ = planned(golden_fact_time())
    rows = neighbor_rows(plan)

    mom, buy, apple = ids[("妈妈", "E")], ids[("买", "P")], ids[("苹果", "E")]
    assert rows[(mom, "N")] == tuple(sorted((buy, apple)))
    assert rows[(apple, "N")] == tuple(sorted((mom, buy)))


def test_predicate_splits_into_s_and_o_rows():
    plan, ids, _ = planned(golden_fact_time())
    rows = neighbor_rows(plan)

    buy, mom, apple = ids[("买", "P")], ids[("妈妈", "E")], ids[("苹果", "E")]
    assert rows[(buy, "S")] == (mom,)
    assert rows[(buy, "O")] == (apple,)
    assert (buy, "N") not in rows, "P must never write an N row"


def test_modifier_writes_cooccurrence():
    plan, ids, _ = planned(golden_fact_time())

    keys = {(write.atom_id, write.anchor_id) for write in plan.cooccurrences}
    for literal in (("昨天", "E"), ("在超市", "E")):
        assert (ids[literal], 904) in keys


def test_modifier_never_enters_neighbor():
    plan, ids, _ = planned(golden_fact_time())
    modifiers = {ids[("昨天", "E")], ids[("在超市", "E")]}

    assert all(write.atom_id not in modifiers for write in plan.neighbors)
    assert all(bit not in write.neighbor_atom_ids for write in plan.neighbors for bit in modifiers)


def test_multi_agent_all_enter_p_s():
    component = fact(
        [
            atom(1, "张三", "E", "agent", 3),
            atom(2, "李四", "E", "agent", 3),
            atom(3, "搬", "P", "predicate", None),
            atom(4, "桌子", "E", "patient", 3),
            atom(5, "椅子", "E", "patient", 3),
        ]
    )
    plan, ids, _ = planned(component)
    rows = neighbor_rows(plan)

    move = ids[("搬", "P")]
    assert rows[(move, "S")] == tuple(sorted((ids[("张三", "E")], ids[("李四", "E")])))


def test_multi_patient_all_enter_p_o_without_truncation():
    component = fact(
        [
            atom(1, "张三", "E", "agent", 3),
            atom(2, "李四", "E", "agent", 3),
            atom(3, "搬", "P", "predicate", None),
            atom(4, "桌子", "E", "patient", 3),
            atom(5, "椅子", "E", "patient", 3),
        ]
    )
    plan, ids, _ = planned(component)
    rows = neighbor_rows(plan)

    move = ids[("搬", "P")]
    assert rows[(move, "O")] == tuple(sorted((ids[("桌子", "E")], ids[("椅子", "E")])))


def test_e_neighbor_collects_other_core_atoms_of_its_frame():
    component = fact(
        [
            atom(1, "张三", "E", "agent", 3),
            atom(2, "李四", "E", "agent", 3),
            atom(3, "搬", "P", "predicate", None),
            atom(4, "桌子", "E", "patient", 3),
            atom(5, "椅子", "E", "patient", 3),
        ]
    )
    plan, ids, _ = planned(component)
    rows = neighbor_rows(plan)

    zhang = ids[("张三", "E")]
    others = tuple(sorted(ids[lit] for lit in (("李四", "E"), ("搬", "P"), ("桌子", "E"), ("椅子", "E"))))
    assert rows[(zhang, "N")] == others


def test_duplicate_neighbor_atom_id_is_deduplicated():
    plan, ids, _ = planned(golden_fact_recursive())
    rows = neighbor_rows(plan)

    ming, zou = ids[("小明", "E")], ids[("揍", "P")]
    assert rows[(ming, "N")] == (zou,)
    assert rows[(zou, "S")] == (ming,)
    assert rows[(zou, "O")] == (ming,)


def test_self_loop_is_never_written_for_e_source():
    plan, ids, _ = planned(golden_fact_recursive())
    rows = neighbor_rows(plan)

    ming = ids[("小明", "E")]
    assert ming not in rows[(ming, "N")], "E source must not include its own atom_id"


def test_same_atom_anchor_cooccurrence_is_deduplicated():
    plan, ids, _ = planned(golden_fact_recursive())

    ming = ids[("小明", "E")]
    ming_writes = [write for write in plan.cooccurrences if write.atom_id == ming]
    assert len(ming_writes) == 1, "two 小明 occurrences in one frame share one anchor bucket"


def test_same_atom_in_two_frames_gets_two_cooccurrence_buckets():
    component = fact(
        [
            atom(1, "妈妈", "E", "agent", 2),
            atom(2, "让", "P", "predicate", None),
            atom(3, "小明", "E", "patient", 2),
            atom(4, "打", "P", "predicate", 3),
            atom(5, "酱油", "E", "patient", 4),
            atom(6, "小明", "E", "patient", 4),
        ]
    )
    plan, ids, _ = planned(component)

    ming = ids[("小明", "E")]
    keys = {(write.atom_id, write.anchor_id) for write in plan.cooccurrences if write.atom_id == ming}
    assert keys == {(ming, 902), (ming, 904)}


def test_output_rows_and_neighbor_ids_are_stably_sorted():
    plan, _, _ = planned(golden_fact_time())

    cooccurrence_keys = [(write.atom_id, write.anchor_id) for write in plan.cooccurrences]
    assert cooccurrence_keys == sorted(cooccurrence_keys)
    neighbor_keys = [(write.atom_id, write.anchor_id, write.role_type) for write in plan.neighbors]
    assert neighbor_keys == sorted(neighbor_keys)
    for write in plan.neighbors:
        assert write.neighbor_atom_ids == tuple(sorted(write.neighbor_atom_ids))


def test_plan_is_independent_of_atom_input_order():
    atoms = [
        atom(1, "妈妈", "E", "agent", 2),
        atom(2, "让", "P", "predicate", None),
        atom(3, "小明", "E", "patient", 2),
        atom(4, "打", "P", "predicate", 3),
        atom(5, "酱油", "E", "patient", 4),
    ]
    forward, forward_ids, _ = planned(fact(atoms))
    reversed_input, reversed_ids, _ = planned(fact(list(reversed(atoms))))

    assert forward == reversed_input
    assert forward_ids == reversed_ids
    assert forward.cooccurrences == tuple(sorted(forward.cooccurrences, key=lambda w: (w.atom_id, w.anchor_id)))


def _stub_atoms():
    return [
        SimpleNamespace(pos=1, text="妈妈", type="E", role="agent", target_occ=2),
        SimpleNamespace(pos=2, text="买", type="P", role="predicate", target_occ=None),
        SimpleNamespace(pos=3, text="苹果", type="E", role="patient", target_occ=2),
    ]


def _stub_plan():
    return AnchorPlan(
        frames={2: PredicateFrame(2, "妈妈 买 苹果")},
        occurrences=(
            OccurrenceRoute(1, ("妈妈", "E"), 2),
            OccurrenceRoute(2, ("买", "P"), 2),
            OccurrenceRoute(3, ("苹果", "E"), 2),
        ),
        semantic_members=(
            SemanticFrameMember(1, 2, "agent", 2, False),
            SemanticFrameMember(2, 2, "predicate", 2, False),
            SemanticFrameMember(3, 2, "patient", 2, False),
        ),
        context_texts=("妈妈 买 苹果",),
    )


_STUB_ATOM_IDS = {("妈妈", "E"): 101, ("买", "P"): 102, ("苹果", "E"): 103}
_STUB_ROUTES = {(101, 2): 9001, (102, 2): 9002, (103, 2): 9003}


def test_missing_atom_id_fails_closed():
    with pytest.raises(BitmapPlanError, match="atom_id"):
        plan_bitmap_writes(
            event_id=1,
            atoms=_stub_atoms(),
            anchor_plan=_stub_plan(),
            atom_ids={("妈妈", "E"): 101, ("买", "P"): 102},
            anchor_routes=_STUB_ROUTES,
        )


def test_missing_anchor_route_fails_closed():
    routes = {key: value for key, value in _STUB_ROUTES.items() if key != (103, 2)}
    with pytest.raises(BitmapPlanError, match="anchor"):
        plan_bitmap_writes(
            event_id=1,
            atoms=_stub_atoms(),
            anchor_plan=_stub_plan(),
            atom_ids=_STUB_ATOM_IDS,
            anchor_routes=routes,
        )


def test_zero_or_null_anchor_id_fails_closed():
    for bad in (0, None):
        routes = dict(_STUB_ROUTES)
        routes[(103, 2)] = bad
        with pytest.raises(BitmapPlanError, match="anchor"):
            plan_bitmap_writes(
                event_id=1,
                atoms=_stub_atoms(),
                anchor_plan=_stub_plan(),
                atom_ids=_STUB_ATOM_IDS,
                anchor_routes=routes,
            )


def test_illegal_role_is_not_silently_swallowed():
    atoms = _stub_atoms()
    atoms[2].role = "subject"
    with pytest.raises(BitmapPlanError, match="role"):
        plan_bitmap_writes(
            event_id=1,
            atoms=atoms,
            anchor_plan=_stub_plan(),
            atom_ids=_STUB_ATOM_IDS,
            anchor_routes=_STUB_ROUTES,
        )


def test_missing_planned_occurrence_fails_closed():
    plan = _stub_plan()
    incomplete = AnchorPlan(
        frames=plan.frames,
        occurrences=plan.occurrences[:-1],
        semantic_members=plan.semantic_members[:-1],
        context_texts=plan.context_texts,
    )
    with pytest.raises(BitmapPlanError, match="missing atom occurrence"):
        plan_bitmap_writes(
            event_id=1,
            atoms=_stub_atoms(),
            anchor_plan=incomplete,
            atom_ids=_STUB_ATOM_IDS,
            anchor_routes=_STUB_ROUTES,
        )


def test_mismatched_planned_literal_fails_closed():
    plan = _stub_plan()
    mismatched = AnchorPlan(
        frames=plan.frames,
        occurrences=(OccurrenceRoute(1, ("别人", "E"), 2), *plan.occurrences[1:]),
        semantic_members=plan.semantic_members,
        context_texts=plan.context_texts,
    )
    with pytest.raises(BitmapPlanError, match="does not match atom"):
        plan_bitmap_writes(
            event_id=1,
            atoms=_stub_atoms(),
            anchor_plan=mismatched,
            atom_ids={**_STUB_ATOM_IDS, ("别人", "E"): 104},
            anchor_routes={**_STUB_ROUTES, (104, 2): 9004},
        )


def test_missing_semantic_members_fail_closed():
    plan = _stub_plan()
    empty = AnchorPlan(
        frames=plan.frames,
        occurrences=plan.occurrences,
        semantic_members=(),
        context_texts=plan.context_texts,
    )
    with pytest.raises(BitmapPlanError, match="semantic member"):
        plan_bitmap_writes(
            event_id=1,
            atoms=_stub_atoms(),
            anchor_plan=empty,
            atom_ids=_STUB_ATOM_IDS,
            anchor_routes=_STUB_ROUTES,
        )


def test_semantic_member_coverage_mismatch_fails_closed():
    plan = _stub_plan()
    missing = AnchorPlan(
        frames=plan.frames,
        occurrences=plan.occurrences,
        semantic_members=plan.semantic_members[:-1],
        context_texts=plan.context_texts,
    )
    with pytest.raises(BitmapPlanError, match="semantic member"):
        plan_bitmap_writes(
            event_id=1,
            atoms=_stub_atoms(),
            anchor_plan=missing,
            atom_ids=_STUB_ATOM_IDS,
            anchor_routes=_STUB_ROUTES,
        )


def test_large_frame_above_50_is_not_truncated():
    atoms = [atom(1, "大动作", "P", "predicate", None)]
    for index in range(25):
        atoms.append(atom(2 + index, f"施事{index}", "E", "agent", 1))
    for index in range(25):
        atoms.append(atom(27 + index, f"受事{index}", "E", "patient", 1))
    plan, ids, _ = planned(fact(atoms))

    assert len(plan.cooccurrences) == 51
    e_rows = [write for write in plan.neighbors if write.role_type == "N"]
    assert len(e_rows) == 50
    assert all(len(write.neighbor_atom_ids) == 50 for write in e_rows)
    s_row = next(write for write in plan.neighbors if write.role_type == "S")
    o_row = next(write for write in plan.neighbors if write.role_type == "O")
    assert len(s_row.neighbor_atom_ids) == 25
    assert len(o_row.neighbor_atom_ids) == 25


# ---------------------------------------------------------------------------
# §12.2 nested predicate-frame isolation
# ---------------------------------------------------------------------------


def test_nested_clause_frames_never_cross_pair():
    plan, ids, _ = planned(golden_fact_recursive())
    rows = neighbor_rows(plan)

    ming, xie, homework, zou = (
        ids[("小明", "E")],
        ids[("没写", "P")],
        ids[("作业", "E")],
        ids[("揍", "P")],
    )
    assert rows[(homework, "N")] == (xie,)
    assert rows[(xie, "O")] == (homework,)
    assert (xie, "S") not in rows, "没写 targets the parent predicate: no semantic agent is borrowed"
    assert rows[(ming, "N")] == (zou,), "without a borrow the same-name E rows stay in their own frame"
    assert homework not in rows[(ming, "N")]
    assert ming not in rows[(homework, "N")]


def test_frozen_mother_lets_child_buy_soy_sauce_isolation():
    component = fact(
        [
            atom(1, "妈妈", "E", "agent", 2),
            atom(2, "让", "P", "predicate", None),
            atom(3, "小明", "E", "patient", 2),
            atom(4, "打", "P", "predicate", 3),
            atom(5, "酱油", "E", "patient", 4),
        ]
    )
    plan, ids, _ = planned(component)
    rows = neighbor_rows(plan)

    mother, rang, ming, beat, sauce = (ids[lit] for lit in (("妈妈", "E"), ("让", "P"), ("小明", "E"), ("打", "P"), ("酱油", "E")))
    assert rows[(rang, "S")] == (mother,)
    assert rows[(rang, "O")] == (ming,)
    assert rows[(beat, "S")] == (ming,), "requirement 08 §8.2: the child frame's S row carries the borrowed pivot"
    assert rows[(beat, "O")] == (sauce,)
    assert rows[(sauce, "N")] == (ming, beat)
    assert rows[(mother, "N")] == (rang, ming)
    # The shared pivot is the one legal overlap node: its own N row unions both
    # frames, while no other node crosses frames.
    assert rows[(ming, "N")] == (mother, rang, beat, sauce)
    assert mother not in rows[(sauce, "N")]
    assert sauce not in rows[(mother, "N")]
    assert set(rows[(rang, "S")]) | set(rows[(rang, "O")]) <= {mother, ming}
    assert set(rows[(beat, "S")]) | set(rows[(beat, "O")]) <= {ming, sauce}


def test_complex_shared_pivot_neighbor_frames():
    component = fact(
        [
            atom(1, "昨天晚上", "E", "modifier", 4),
            atom(2, "妈妈", "E", "agent", 4),
            atom(3, "在厨房", "E", "modifier", 4),
            atom(4, "让", "P", "predicate", None),
            atom(5, "小明", "E", "patient", 4),
            atom(6, "桌上", "E", "modifier", 9),
            atom(7, "两个", "E", "modifier", 9),
            atom(8, "红", "E", "modifier", 9),
            atom(9, "苹果", "E", "patient", 10),
            atom(10, "洗", "P", "predicate", 5),
            atom(11, "干净", "E", "modifier", 10),
        ]
    )
    plan, ids, _ = planned(component)
    rows = neighbor_rows(plan)

    mother, rang, ming, apple, wash = (
        ids[("妈妈", "E")],
        ids[("让", "P")],
        ids[("小明", "E")],
        ids[("苹果", "E")],
        ids[("洗", "P")],
    )
    assert rows[(rang, "S")] == (mother,)
    assert rows[(rang, "O")] == (ming,)
    assert rows[(wash, "S")] == (ming,)
    assert rows[(wash, "O")] == (apple,)
    assert rows[(mother, "N")] == (rang, ming)
    assert rows[(apple, "N")] == (ming, wash)
    assert rows[(ming, "N")] == (mother, rang, apple, wash)
    # Non-shared nodes never cross frames.
    assert set(rows[(rang, "S")]) | set(rows[(rang, "O")]) <= {mother, ming}
    assert set(rows[(wash, "S")]) | set(rows[(wash, "O")]) <= {ming, apple}
    assert mother not in rows[(apple, "N")] and rang not in rows[(apple, "N")]
    assert apple not in rows[(mother, "N")] and wash not in rows[(mother, "N")]
    # Modifiers write no Neighbor rows and never appear as neighbors.
    modifier_ids = {ids[(text, "E")] for text in ("昨天晚上", "在厨房", "桌上", "两个", "红", "干净")}
    assert all(write.atom_id not in modifier_ids for write in plan.neighbors)
    for write in plan.neighbors:
        assert not (set(write.neighbor_atom_ids) & modifier_ids)


def test_nested_cooccurrence_still_writes_every_occurrence():
    component = fact(
        [
            atom(1, "妈妈", "E", "agent", 2),
            atom(2, "让", "P", "predicate", None),
            atom(3, "小明", "E", "patient", 2),
            atom(4, "打", "P", "predicate", 3),
            atom(5, "酱油", "E", "patient", 4),
        ]
    )
    plan, ids, _ = planned(component, event_id=555)

    keys = {(write.atom_id, write.anchor_id) for write in plan.cooccurrences}
    assert keys == {(ids[lit], 902) for lit in (("妈妈", "E"), ("让", "P"), ("小明", "E"))} | {
        (ids[lit], 904) for lit in (("打", "P"), ("酱油", "E"))
    }
    assert {write.event_id for write in plan.cooccurrences} == {555}
    # Requirement 08 §8.1: the borrowed member adds no second Cooccurrence key.
    ming_keys = {(write.atom_id, write.anchor_id) for write in plan.cooccurrences if write.atom_id == ids[("小明", "E")]}
    assert ming_keys == {(ids[("小明", "E")], 902)}


# ---------------------------------------------------------------------------
# §12.3 SQL / upsert behavior against the fake store
# ---------------------------------------------------------------------------


def sample_plan(cooccurrences=(), neighbors=()) -> BitmapWritePlan:
    return BitmapWritePlan(cooccurrences=tuple(cooccurrences), neighbors=tuple(neighbors))


async def test_first_insert_then_on_conflict_or_is_idempotent():
    store = FakeStore()
    plan = sample_plan(
        cooccurrences=(CooccurrenceBitmapWrite(5, 9, 881),),
        neighbors=(NeighborBitmapWrite(5, 9, "N", (10, 11)), NeighborBitmapWrite(5, 9, "S", (10,))),
    )
    await apply_bitmap_writes(FakeConn(store), SCHEMA, plan)
    await apply_bitmap_writes(FakeConn(store), SCHEMA, plan)

    assert store.cooccurrence_bitmaps[(5, 9)] == {881}
    assert store.neighbor_bitmaps[(5, 9, "N")] == {10, 11}
    assert store.neighbor_bitmaps[(5, 9, "S")] == {10}


async def test_second_event_bit_is_retained():
    store = FakeStore()
    await apply_bitmap_writes(
        FakeConn(store), SCHEMA, sample_plan(cooccurrences=(CooccurrenceBitmapWrite(5, 9, 881),))
    )
    await apply_bitmap_writes(
        FakeConn(store), SCHEMA, sample_plan(cooccurrences=(CooccurrenceBitmapWrite(5, 9, 882),))
    )

    assert store.cooccurrence_bitmaps[(5, 9)] == {881, 882}


async def test_second_neighbor_bit_is_retained():
    store = FakeStore()
    await apply_bitmap_writes(
        FakeConn(store), SCHEMA, sample_plan(neighbors=(NeighborBitmapWrite(5, 9, "N", (10,)),))
    )
    await apply_bitmap_writes(
        FakeConn(store), SCHEMA, sample_plan(neighbors=(NeighborBitmapWrite(5, 9, "N", (11,)),))
    )

    assert store.neighbor_bitmaps[(5, 9, "N")] == {10, 11}


async def test_s_o_n_rows_are_isolated():
    store = FakeStore()
    plan = sample_plan(
        neighbors=(
            NeighborBitmapWrite(5, 9, "N", (10,)),
            NeighborBitmapWrite(5, 9, "S", (10,)),
            NeighborBitmapWrite(5, 9, "O", (11,)),
        )
    )
    await apply_bitmap_writes(FakeConn(store), SCHEMA, plan)

    assert store.neighbor_bitmaps[(5, 9, "N")] == {10}
    assert store.neighbor_bitmaps[(5, 9, "S")] == {10}
    assert store.neighbor_bitmaps[(5, 9, "O")] == {11}


async def test_writer_never_selects_then_updates():
    store = FakeStore()
    plan = sample_plan(
        cooccurrences=(CooccurrenceBitmapWrite(5, 9, 881),),
        neighbors=(NeighborBitmapWrite(5, 9, "N", (10,)),),
    )
    await apply_bitmap_writes(FakeConn(store), SCHEMA, plan)

    calls = bitmap_calls(store)
    assert len(calls) == 2, "one aggregated upsert per target row, never one SQL per bit"
    for sql, _args in calls:
        assert sql.lstrip().upper().startswith("INSERT")
        assert "ON CONFLICT" in sql
        assert "rb64_build" in sql and "rb64_or" in sql
    assert not any("SELECT" in sql for sql, _args in calls)


async def test_writer_preserves_plan_order():
    store = FakeStore()
    plan = sample_plan(
        cooccurrences=(CooccurrenceBitmapWrite(1, 2, 881), CooccurrenceBitmapWrite(3, 4, 881)),
        neighbors=(
            NeighborBitmapWrite(1, 2, "N", (3,)),
            NeighborBitmapWrite(5, 6, "O", (7,)),
        ),
    )
    await apply_bitmap_writes(FakeConn(store), SCHEMA, plan)

    args = [args for _sql, args in bitmap_calls(store)]
    assert [(entry[0], entry[1]) for entry in args] == [(1, 2), (3, 4), (1, 2), (5, 6)]


async def test_writer_wraps_db_errors_and_propagates_cancellation():
    class _BoomConn:
        async def execute(self, sql, *args):
            raise RuntimeError("boom")

    with pytest.raises(BitmapWriteError) as excinfo:
        await apply_bitmap_writes(_BoomConn(), SCHEMA, sample_plan(cooccurrences=(CooccurrenceBitmapWrite(1, 2, 3),)))
    assert isinstance(excinfo.value.__cause__, RuntimeError)

    class _CancelConn:
        async def execute(self, sql, *args):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await apply_bitmap_writes(
            _CancelConn(), SCHEMA, sample_plan(cooccurrences=(CooccurrenceBitmapWrite(1, 2, 3),))
        )


# ---------------------------------------------------------------------------
# End-to-end: bitmaps live inside the fact transaction
# ---------------------------------------------------------------------------


async def test_ingest_writes_bitmaps_inside_the_fact_transaction(monkeypatch):
    store = FakeStore()
    depths: list[int] = []
    original = store.execute

    async def spy(sql, *args):
        if ".cooccurrence_bitmaps" in sql or ".neighbor_bitmaps" in sql:
            depths.append(store._tx_depth)
        return await original(sql, *args)

    store.execute = spy
    await run_ingest(store, monkeypatch, [golden_fact_time()])

    event_id = next(iter(store.events.values()))["event_id"]
    rows = [args for kind, sql, args in store.calls if kind == "execute" and ".event_atoms" in sql]
    assert rows and all(len(row) == 6 for row in rows)  # (event, occ, atom, role, target, anchor)

    # Cooccurrence: every event_atom (including the two modifiers) in a bucket.
    assert store.cooccurrence_bitmaps == {(row[2], row[5]): {event_id} for row in rows}
    # Neighbor: derived from the very rows written to event_atoms.
    core = [row for row in rows if row[3] in ("A", "P", "R")]
    agents = {row[2] for row in core if row[3] == "A"}
    patients = {row[2] for row in core if row[3] == "P"}
    expected: dict[tuple[int, int, str], set[int]] = {}
    for row in core:
        atom_id, anchor_id, role = row[2], row[5], row[3]
        if role == "R":
            if agents - {atom_id}:
                expected[(atom_id, anchor_id, "S")] = agents - {atom_id}
            if patients - {atom_id}:
                expected[(atom_id, anchor_id, "O")] = patients - {atom_id}
        else:
            others = {other[2] for other in core if other[2] != atom_id}
            if others:
                expected[(atom_id, anchor_id, "N")] = others
    assert store.neighbor_bitmaps == expected

    assert depths and all(depth > 0 for depth in depths), "bitmap writes must run inside the open transaction"
    event_atom_positions = [
        index for index, call in enumerate(store.calls)
        if call[0] == "execute" and ".event_atoms" in call[1]
    ]
    bitmap_positions = [
        index for index, call in enumerate(store.calls)
        if call[0] == "execute" and (".cooccurrence_bitmaps" in call[1] or ".neighbor_bitmaps" in call[1])
    ]
    assert min(bitmap_positions) > max(event_atom_positions)


async def test_bitmap_upserts_follow_stable_key_order(monkeypatch):
    store = FakeStore()
    await run_ingest(store, monkeypatch, [golden_fact_time()])

    keys = []
    for sql, args in bitmap_calls(store):
        if ".cooccurrence_bitmaps" in sql:
            keys.append(("co", args[0], args[1], ""))
        else:
            keys.append(("nb", args[0], args[1], args[2]))
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# §12.4 failure injection: zero half-state across fact + anchors + bitmaps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fail_after",
    [
        {".cooccurrence_bitmaps": 0},  # first cooccurrence write
        {".cooccurrence_bitmaps": 1},  # middle cooccurrence write
        {".neighbor_bitmaps": 0},  # first neighbor write
        {".neighbor_bitmaps": 1},  # middle neighbor write
    ],
)
async def test_bitmap_write_failures_roll_back_whole_component(monkeypatch, fail_after):
    store = FakeStore()
    store.fail_after = dict(fail_after)
    await run_ingest(store, monkeypatch, [golden_fact_time()])

    _assert_no_fact_state(store)
    assert store.rollbacks >= 1
    alerts = store.alerts_by_code("event_ingest_failed")
    assert len(alerts) == 1
    assert alerts[0]["stage"] == "event_ingest"
    assert alerts[0]["severity"] == "error"


async def test_failure_after_all_bitmap_writes_before_commit_rolls_back(monkeypatch):
    store = FakeStore()
    original = store.execute
    state = {"bitmap_writes": 0, "total": 0}

    async def spy(sql, *args):
        result = await original(sql, *args)
        if ".cooccurrence_bitmaps" in sql or ".neighbor_bitmaps" in sql:
            state["bitmap_writes"] += 1
            state["total"] += 1
            if state["bitmap_writes"] == state.get("expected", 9):
                raise RuntimeError("injected after the final bitmap write")
        return result

    store.execute = spy
    expected = 5 + 4  # golden_fact_time: 5 cooccurrence rows + 4 neighbor rows
    state["expected"] = expected
    await run_ingest(store, monkeypatch, [golden_fact_time()])

    assert state["total"] == expected, "all bitmap writes must have been attempted"
    _assert_no_fact_state(store)
    assert store.rollbacks >= 1
    assert len(store.alerts_by_code("event_ingest_failed")) == 1


async def test_commit_failure_rolls_back_everything(monkeypatch):
    store = FakeStore()
    store.fail_on_commit = True
    await run_ingest(store, monkeypatch, [golden_fact_time()])

    _assert_no_fact_state(store)
    assert store.rollbacks >= 1
    assert len(store.alerts_by_code("event_ingest_failed")) == 1


async def test_bitmap_failure_rolls_back_anchor_ema(monkeypatch):
    """§9.2: a bitmap failure rolls back the create/EMA anchor changes of the
    same component — the committed buckets stay bit-for-bit untouched."""
    store = FakeStore()
    await run_ingest(store, monkeypatch, [golden_fact_time()])
    committed = {
        anchor["anchor_id"]: (anchor["centroid"], anchor["total_count"]) for anchor in store.anchors_rows()
    }
    committed_cooccurrence = {
        key: set(bucket) for key, bucket in store.cooccurrence_bitmaps.items()
    }
    committed_neighbor = {key: set(bucket) for key, bucket in store.neighbor_bitmaps.items()}
    assert committed and all(count == 1 for _, count in committed.values())

    # The identical component reuses every anchor (EMA would bump total_count
    # to 2) and then fails on the very first bitmap write.
    store.fail_after = {".cooccurrence_bitmaps": 0}
    await run_ingest(store, monkeypatch, [golden_fact_time()])

    assert len(store.events) == 1  # only the first, committed fact
    assert {
        anchor["anchor_id"]: (anchor["centroid"], anchor["total_count"]) for anchor in store.anchors_rows()
    } == committed
    assert {key: set(bucket) for key, bucket in store.cooccurrence_bitmaps.items()} == committed_cooccurrence
    assert {key: set(bucket) for key, bucket in store.neighbor_bitmaps.items()} == committed_neighbor
    assert len(store.alerts_by_code("event_ingest_failed")) == 1


# ---------------------------------------------------------------------------
# §12.5 concurrent union behavior
# ---------------------------------------------------------------------------


async def test_concurrent_upserts_union_bits_without_lost_update():
    store = FakeStore()

    async def yield_once():
        await asyncio.sleep(0)

    store.interleave = yield_once
    plan_a = sample_plan(
        cooccurrences=(CooccurrenceBitmapWrite(5, 9, 881),),
        neighbors=(NeighborBitmapWrite(5, 9, "N", (10,)),),
    )
    plan_b = sample_plan(
        cooccurrences=(CooccurrenceBitmapWrite(5, 9, 882),),
        neighbors=(NeighborBitmapWrite(5, 9, "N", (11,)),),
    )
    await asyncio.gather(
        apply_bitmap_writes(FakeConn(store), SCHEMA, plan_a),
        apply_bitmap_writes(FakeConn(store), SCHEMA, plan_b),
    )

    assert store.cooccurrence_bitmaps[(5, 9)] == {881, 882}
    assert store.neighbor_bitmaps[(5, 9, "N")] == {10, 11}


async def test_concurrent_components_keep_both_event_bits(monkeypatch):
    import datetime

    store = FakeStore()

    async def yield_once():
        await asyncio.sleep(0)

    store.interleave = yield_once

    def fake_extract(text, *, extract_once):
        return outcome([golden_fact_time()])

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", fake_extract)

    async def run_once(offset):
        await ingest_noesis_batch(
            [{"content": CONTENT, "event_date": OBSERVED_AT + datetime.timedelta(days=offset)}],
            "bank-1",
            noesis_config(),
            llm_config=llm_config(),
            extract_once_factory=FakeExtractOnceFactory(),
            pool_factory=pool_factory_for(store),
            embedding_client_factory=embedding_factory_for(FakeEmbeddingClient()),
        )

    await asyncio.gather(run_once(0), run_once(1))

    assert len(store.events) == 2
    event_ids = {event["event_id"] for event in store.events.values()}
    assert len(event_ids) == 2
    # The cooperative fake may interleave anchor creation into two buckets for
    # one atom; either way the per-atom event-bit union keeps both events and
    # every neighbor row keeps its bit (the real single-statement rb64_or
    # union is proven against PostgreSQL in the remote smoke).
    per_atom_events: dict[int, set[int]] = {}
    for (atom_id, _anchor_id), bucket in store.cooccurrence_bitmaps.items():
        per_atom_events.setdefault(atom_id, set()).update(bucket)
    assert all(bits == event_ids for bits in per_atom_events.values())
    assert store.neighbor_bitmaps and all(len(bucket) >= 1 for bucket in store.neighbor_bitmaps.values())
