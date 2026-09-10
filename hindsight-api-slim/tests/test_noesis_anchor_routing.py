"""Noesis anchor routing tests: predicate frames (§13.1) and the anchor algorithm (§13.3).

Requirement 05 contract coverage, fully offline (FakeStore / FakeEmbeddingClient).
The §13.1 group freezes the pure predicate-frame construction: context texts,
occurrence assignment, and broken-closure failures, including the frozen
"妈妈让小明打酱油" example (§5.5). The §13.3 group freezes the anchor routing
algorithm against the in-memory store: reuse threshold, nearest-active
selection, deterministic tie-break, EMA, per-frame sampling, status filtering,
and the no-support_count / no-NULL-anchor SQL contract.
"""

from __future__ import annotations

import pytest
from hyperextract.noesis import FactComponent

from hindsight_api.engine.retain.noesis_anchor import (
    REUSE_MAX_DISTANCE,
    AnchorFrameError,
    AnchorPlan,
    AnchorRouteError,
    OccurrenceRoute,
    PredicateFrame,
    assign_occurrence_frames,
    build_predicate_frames,
    ema_centroid,
    parse_centroid_text,
    plan_anchor_routing,
    prepare_context_vectors,
    route_anchors,
    should_reuse,
)
from hindsight_api.engine.retain.noesis_embedding import EmbeddingError
from tests.noesis_fakes import (
    FakeConn,
    FakeEmbeddingClient,
    FakeStore,
    fake_context_vector,
    golden_fact_recursive,
    golden_fact_time,
)

SCHEMA = "noesis_core"
DIMENSION = 1024


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def atom(pos, text, type_, role, target):
    return {"pos": pos, "text": text, "type": type_, "role": role, "target_occ": target, "resolved": None}


def fact(atoms):
    """FactComponent from atom dicts. The tree mirrors the root frame only —
    the frame algorithms read atoms, the tree just feeds the stored event JSON."""
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


def basis(index, dimension=DIMENSION):
    """Unit basis vector: distinct indices are orthogonal (cosine distance 1.0)."""
    vector = [0.0] * dimension
    vector[index] = 1.0
    return tuple(vector)


def seed_atom(store, text, atom_type):
    store._ids["atom"] += 1
    atom_id = store._ids["atom"]
    store.atoms[(text, atom_type)] = {"atom_id": atom_id, "embedding": None}
    return atom_id


def make_plan(frames, occurrences):
    """Hand-built AnchorPlan: ``frames`` is {predicate_pos: context_text},
    ``occurrences`` is [(pos, text, atom_type, frame_pos)]. plan_anchor_routing
    is still a skeleton, so the §13.3 tests construct plans directly."""
    ordered = dict(sorted(frames.items()))
    texts: list[str] = []
    for text in ordered.values():
        if text not in texts:
            texts.append(text)
    return AnchorPlan(
        frames={pos: PredicateFrame(pos, text) for pos, text in ordered.items()},
        occurrences=tuple(
            OccurrenceRoute(pos, (text, atom_type), frame_pos) for pos, text, atom_type, frame_pos in occurrences
        ),
        context_texts=tuple(texts),
    )


async def route(store, *, atom_ids, plan, context_vectors):
    return await route_anchors(FakeConn(store), SCHEMA, atom_ids=atom_ids, plan=plan, context_vectors=context_vectors)


def anchor_sql_calls(store):
    return [sql for _kind, sql, _args in store.calls if ".anchors" in sql]


# ---------------------------------------------------------------------------
# §13.1 predicate-frame construction (pure units)
# ---------------------------------------------------------------------------


def test_frame_single_agent_predicate_patient():
    atoms = [
        atom(1, "妈妈", "E", "agent", 2),
        atom(2, "买", "P", "predicate", None),
        atom(3, "苹果", "E", "patient", 2),
    ]
    assert build_predicate_frames(fact(atoms).atoms) == {2: PredicateFrame(2, "妈妈 买 苹果")}


def test_frame_predicate_only():
    atoms = [atom(1, "下雨", "P", "predicate", None)]
    assert build_predicate_frames(fact(atoms).atoms) == {1: PredicateFrame(1, "下雨")}


def test_frame_agent_predicate_no_patient():
    atoms = [
        atom(1, "张三", "E", "agent", 2),
        atom(2, "跑步", "P", "predicate", None),
    ]
    assert build_predicate_frames(fact(atoms).atoms) == {2: PredicateFrame(2, "张三 跑步")}


def test_frame_multi_agent_multi_patient_pos_order():
    # "甲和乙共同修复服务器与数据库" — every core role kept, pos-stable join.
    atoms = [
        atom(1, "甲", "E", "agent", 5),
        atom(2, "乙", "E", "agent", 5),
        atom(3, "服务器", "E", "patient", 5),
        atom(4, "数据库", "E", "patient", 5),
        atom(5, "修复", "P", "predicate", None),
    ]
    assert build_predicate_frames(fact(atoms).atoms) == {5: PredicateFrame(5, "甲 乙 修复 服务器 数据库")}


def test_frame_modifier_excluded_from_context_text():
    # golden: 昨天/在超市 modify 买 and never enter the context text.
    assert build_predicate_frames(golden_fact_time().atoms) == {4: PredicateFrame(4, "妈妈 买 苹果")}


def test_modifier_pointing_at_predicate_assigned():
    atoms = [
        atom(1, "昨天", "E", "modifier", 2),
        atom(2, "买", "P", "predicate", None),
        atom(3, "苹果", "E", "patient", 2),
    ]
    component = fact(atoms)
    frames = build_predicate_frames(component.atoms)
    assert frames == {2: PredicateFrame(2, "买 苹果")}
    assert assign_occurrence_frames(component.atoms, frames) == {1: 2, 2: 2, 3: 2}


def test_modifier_chain_through_core_roles():
    # 昨天 → agent 妈妈 → predicate 买; 很 → patient 苹果 → predicate 买.
    atoms = [
        atom(1, "昨天", "E", "modifier", 2),
        atom(2, "妈妈", "E", "agent", 4),
        atom(3, "很", "E", "modifier", 5),
        atom(4, "买", "P", "predicate", None),
        atom(5, "苹果", "E", "patient", 4),
    ]
    component = fact(atoms)
    frames = build_predicate_frames(component.atoms)
    assert frames == {4: PredicateFrame(4, "妈妈 买 苹果")}
    assert assign_occurrence_frames(component.atoms, frames) == {1: 4, 2: 4, 3: 4, 4: 4, 5: 4}


def test_root_and_nested_clause_two_frames():
    atoms = [
        atom(1, "老师", "E", "agent", 2),
        atom(2, "说", "P", "predicate", None),
        atom(3, "学生", "E", "patient", 2),
        atom(4, "写", "P", "predicate", 3),  # nested clause predicate → parent patient
        atom(5, "作业", "E", "patient", 4),
    ]
    component = fact(atoms)
    frames = build_predicate_frames(component.atoms)
    assert frames == {2: PredicateFrame(2, "老师 说 学生"), 4: PredicateFrame(4, "写 作业")}
    assert assign_occurrence_frames(component.atoms, frames) == {1: 2, 2: 2, 3: 2, 4: 4, 5: 4}


def test_deeply_nested_clause_frames():
    atoms = [
        atom(1, "甲", "E", "agent", 2),
        atom(2, "说", "P", "predicate", None),
        atom(3, "乙", "E", "patient", 2),
        atom(4, "告诉", "P", "predicate", 3),  # nested 1 → patient 乙
        atom(5, "丙", "E", "patient", 4),
        atom(6, "走", "P", "predicate", 5),  # nested 2 → patient 丙
    ]
    frames = build_predicate_frames(fact(atoms).atoms)
    assert frames == {
        2: PredicateFrame(2, "甲 说 乙"),
        4: PredicateFrame(4, "告诉 丙"),
        6: PredicateFrame(6, "走"),
    }


def test_nested_clause_without_explicit_agent_no_implied_injection():
    # golden_fact_recursive: the nested clause predicate 没写(2) has no explicit
    # agent — 小明(1) is the ROOT agent (target=4) and 小明(5) the root patient
    # (resolved=True, still an ordinary atoms entry). The implied agent recorded
    # in tree.nested[0] must NOT be injected into the clause frame.
    frames = build_predicate_frames(golden_fact_recursive().atoms)
    assert frames == {
        2: PredicateFrame(2, "没写 作业"),
        4: PredicateFrame(4, "小明 揍 小明"),
    }


def test_duplicate_context_text_once_in_plan():
    atoms = [
        atom(1, "打", "P", "predicate", None),
        atom(2, "酱油", "E", "patient", 1),
        atom(3, "打", "P", "predicate", 1),  # second clause, identical frame text
        atom(4, "酱油", "E", "patient", 3),
    ]
    plan = plan_anchor_routing(fact(atoms).atoms)
    assert plan.context_texts == ("打 酱油",)  # identical texts embedded once


def test_missing_target_pos_fails():
    atoms = [
        atom(1, "妈妈", "E", "agent", 9),  # pos 9 does not exist
        atom(2, "买", "P", "predicate", None),
        atom(3, "苹果", "E", "patient", 2),
    ]
    with pytest.raises(AnchorFrameError):
        build_predicate_frames(fact(atoms).atoms)


def test_target_cycle_fails():
    atoms = [
        atom(1, "很", "E", "modifier", 2),
        atom(2, "快", "E", "modifier", 1),  # 1 ↔ 2 cycle
        atom(3, "跑", "P", "predicate", None),
    ]
    with pytest.raises(AnchorFrameError):
        build_predicate_frames(fact(atoms).atoms)


def test_multiple_root_predicates_fail_closed():
    atoms = [
        atom(1, "小明", "E", "agent", 2),
        atom(2, "吃", "P", "predicate", None),
        atom(3, "苹果", "E", "patient", 2),
        atom(4, "玩", "P", "predicate", None),
        atom(5, "手机", "E", "patient", 4),
    ]
    with pytest.raises(AnchorFrameError, match="expected exactly one"):
        build_predicate_frames(fact(atoms).atoms)


def test_modifier_never_reaches_predicate_fails():
    atoms = [
        atom(1, "据说", "E", "modifier", None),  # chain ends without reaching a predicate
        atom(2, "买", "P", "predicate", None),
        atom(3, "苹果", "E", "patient", 2),
    ]
    with pytest.raises(AnchorFrameError):
        build_predicate_frames(fact(atoms).atoms)


def test_shuffled_json_array_pos_order_stable():
    shuffled = [
        atom(5, "苹果", "E", "patient", 4),
        atom(1, "昨天", "E", "modifier", 4),
        atom(4, "买", "P", "predicate", None),
        atom(3, "在超市", "E", "modifier", 4),
        atom(2, "妈妈", "E", "agent", 4),
    ]
    expected = build_predicate_frames(golden_fact_time().atoms)
    assert build_predicate_frames(fact(shuffled).atoms) == expected
    assert expected == {4: PredicateFrame(4, "妈妈 买 苹果")}


def test_frozen_example_mother_lets_child_buy_soy_sauce():
    # Requirement 05 §5.5: 妈妈让小明打酱油 — exactly two frames; 打/酱油 never
    # inherit the root frame "妈妈 让 小明" and never the whole sentence.
    atoms = [
        atom(1, "妈妈", "E", "agent", 2),
        atom(2, "让", "P", "predicate", None),
        atom(3, "小明", "E", "patient", 2),
        atom(4, "打", "P", "predicate", 2),
        atom(5, "酱油", "E", "patient", 4),
    ]
    component = fact(atoms)
    frames = build_predicate_frames(component.atoms)
    assert frames == {2: PredicateFrame(2, "妈妈 让 小明"), 4: PredicateFrame(4, "打 酱油")}
    assignment = assign_occurrence_frames(component.atoms, frames)
    assert assignment == {1: 2, 2: 2, 3: 2, 4: 4, 5: 4}  # 妈妈/让/小明→2, 打/酱油→4
    plan = plan_anchor_routing(component.atoms)
    assert plan.occurrences == (
        OccurrenceRoute(1, ("妈妈", "E"), 2),
        OccurrenceRoute(2, ("让", "P"), 2),
        OccurrenceRoute(3, ("小明", "E"), 2),
        OccurrenceRoute(4, ("打", "P"), 4),
        OccurrenceRoute(5, ("酱油", "E"), 4),
    )
    assert plan.context_texts == ("妈妈 让 小明", "打 酱油")


# ---------------------------------------------------------------------------
# §13.3 anchor routing algorithm
# ---------------------------------------------------------------------------


async def test_no_active_anchor_creates_with_count_one():
    store = FakeStore()
    atom_id = seed_atom(store, "苹果", "E")
    plan = make_plan({4: "妈妈 买 苹果"}, [(5, "苹果", "E", 4)])

    routes = await route(
        store, atom_ids={("苹果", "E"): atom_id}, plan=plan, context_vectors={"妈妈 买 苹果": list(basis(0))}
    )

    assert set(routes) == {(atom_id, 4)}
    anchor_id = routes[(atom_id, 4)]
    rows = store.anchors_rows()
    assert len(rows) == 1
    assert rows[0]["anchor_id"] == anchor_id
    assert rows[0]["atom_id"] == atom_id
    assert rows[0]["centroid"] == basis(0)  # initial centroid IS the context vector
    assert rows[0]["total_count"] == 1
    assert rows[0]["status"] == "A"


def test_reuse_gate_is_cosine_distance_not_similarity():
    assert REUSE_MAX_DISTANCE == 0.25  # cosine DISTANCE; 0.75 is never a distance gate
    assert should_reuse(REUSE_MAX_DISTANCE) is True
    assert should_reuse(0.0) is True
    assert should_reuse(0.75) is False  # 0.75 as a distance is far beyond the gate


def test_cosine_distance_exactly_threshold_reuses():
    # cosine distance exactly 0.25 (similarity exactly 0.75) is a reuse — the
    # boundary is inclusive; asserted on the pure gate to avoid fp jitter.
    assert should_reuse(0.25) is True
    assert should_reuse(0.24999999999999997) is True  # the double just below 0.25


async def test_distance_slightly_above_threshold_creates_new():
    assert should_reuse(0.25 + 1e-9) is False

    store = FakeStore()
    atom_id = seed_atom(store, "苹果", "E")
    existing = store.seed_anchor(atom_id, basis(1), total_count=5)
    plan = make_plan({4: "妈妈 买 苹果"}, [(5, "苹果", "E", 4)])

    routes = await route(
        store, atom_ids={("苹果", "E"): atom_id}, plan=plan, context_vectors={"妈妈 买 苹果": list(basis(0))}
    )

    # orthogonal context (cosine distance 1.0) → a new anchor; existing untouched
    assert routes[(atom_id, 4)] != existing
    assert len(store.anchors_rows()) == 2
    assert store.anchors[existing]["total_count"] == 5


async def test_multiple_anchors_nearest_active_wins():
    store = FakeStore()
    atom_id = seed_atom(store, "苹果", "E")
    far = store.seed_anchor(atom_id, basis(1), total_count=3)
    near = store.seed_anchor(atom_id, basis(0), total_count=3)
    plan = make_plan({4: "妈妈 买 苹果"}, [(5, "苹果", "E", 4)])

    routes = await route(
        store, atom_ids={("苹果", "E"): atom_id}, plan=plan, context_vectors={"妈妈 买 苹果": list(basis(0))}
    )

    assert routes[(atom_id, 4)] == near
    assert store.anchors[near]["total_count"] == 4  # EMA hit on the nearest
    assert store.anchors[far]["total_count"] == 3  # untouched


async def test_distance_tie_smallest_anchor_id_wins():
    store = FakeStore()
    atom_id = seed_atom(store, "苹果", "E")
    first = store.seed_anchor(atom_id, basis(0), total_count=2)
    second = store.seed_anchor(atom_id, basis(0), total_count=2)
    plan = make_plan({4: "妈妈 买 苹果"}, [(5, "苹果", "E", 4)])

    routes = await route(
        store, atom_ids={("苹果", "E"): atom_id}, plan=plan, context_vectors={"妈妈 买 苹果": list(basis(0))}
    )

    assert first < second  # seeded ascending
    assert routes[(atom_id, 4)] == first  # deterministic anchor_id ASC tie-break
    assert store.anchors[first]["total_count"] == 3
    assert store.anchors[second]["total_count"] == 2


async def test_dormant_and_merged_status_excluded_from_routing():
    store = FakeStore()
    atom_id = seed_atom(store, "苹果", "E")
    dormant = store.seed_anchor(atom_id, basis(0), status="D")
    merged = store.seed_anchor(atom_id, basis(0), status="M")
    plan = make_plan({4: "妈妈 买 苹果"}, [(5, "苹果", "E", 4)])

    routes = await route(
        store, atom_ids={("苹果", "E"): atom_id}, plan=plan, context_vectors={"妈妈 买 苹果": list(basis(0))}
    )

    anchor_id = routes[(atom_id, 4)]
    assert anchor_id not in (dormant, merged)  # all inactive → a fresh anchor
    row = store.anchors[anchor_id]
    assert row["status"] == "A"
    assert row["total_count"] == 1
    assert store.anchors[dormant]["total_count"] == 0
    assert store.anchors[merged]["total_count"] == 0


def test_ema_centroid_exact_weights():
    old = [10.0] * DIMENSION
    context = [20.0] * DIMENSION
    assert ema_centroid(old, context) == [0.9 * 10.0 + 0.1 * 20.0] * DIMENSION
    assert ema_centroid([0.0] * DIMENSION, [1.0] * DIMENSION) == [0.1] * DIMENSION

    with pytest.raises(AnchorRouteError):
        ema_centroid([0.0] * 512, [0.0] * 512)  # wrong dimension


async def test_hit_increments_total_count():
    store = FakeStore()
    atom_id = seed_atom(store, "苹果", "E")
    anchor_id = store.seed_anchor(atom_id, basis(0), total_count=5)
    plan = make_plan({4: "妈妈 买 苹果"}, [(5, "苹果", "E", 4)])

    routes = await route(
        store, atom_ids={("苹果", "E"): atom_id}, plan=plan, context_vectors={"妈妈 买 苹果": list(basis(0))}
    )

    assert routes[(atom_id, 4)] == anchor_id  # same-direction context hits
    assert store.anchors[anchor_id]["total_count"] == 6


async def test_create_writes_total_count_one_directly():
    store = FakeStore()
    atom_id = seed_atom(store, "苹果", "E")
    plan = make_plan({4: "妈妈 买 苹果"}, [(5, "苹果", "E", 4)])

    await route(store, atom_ids={("苹果", "E"): atom_id}, plan=plan, context_vectors={"妈妈 买 苹果": list(basis(0))})

    inserts = [
        sql
        for kind, sql, _args in store.calls
        if kind == "fetchrow" and ".anchors" in sql and sql.lstrip().upper().startswith("INSERT")
    ]
    assert len(inserts) == 1  # one insert, no count-0 pre-write
    assert ", 1, 'A')" in inserts[0]  # explicit total_count=1 in the VALUES clause
    assert store.anchors_rows()[0]["total_count"] == 1


def test_bad_centroid_fails_without_zero_vector():
    with pytest.raises(AnchorRouteError):
        parse_centroid_text("[1,2]")  # wrong dimension
    with pytest.raises(AnchorRouteError):
        parse_centroid_text("not-a-vector")  # malformed
    nonfinite = "[" + ",".join(["nan"] + ["0.0"] * (DIMENSION - 1)) + "]"
    with pytest.raises(AnchorRouteError):
        parse_centroid_text(nonfinite)  # non-finite component at the right dimension
    good = "[" + ",".join(["0.5"] * DIMENSION) + "]"
    assert parse_centroid_text(good) == [0.5] * DIMENSION


async def test_e_and_p_literals_share_one_route_call():
    store = FakeStore()
    apple = seed_atom(store, "苹果", "E")
    buy = seed_atom(store, "买", "P")
    plan = make_plan({4: "妈妈 买 苹果"}, [(2, "买", "P", 4), (5, "苹果", "E", 4)])

    routes = await route(
        store,
        atom_ids={("苹果", "E"): apple, ("买", "P"): buy},
        plan=plan,
        context_vectors={"妈妈 买 苹果": list(basis(0))},
    )

    assert set(routes) == {(apple, 4), (buy, 4)}
    assert all(anchor_id is not None for anchor_id in routes.values())
    assert store.locked_atoms == sorted(store.locked_atoms)  # ascending atom_id lock order


async def test_same_atom_same_frame_sampled_once():
    store = FakeStore()
    apple = seed_atom(store, "苹果", "E")
    # two occurrences of the same typed literal in the same frame (§5.6)
    plan = make_plan({4: "妈妈 买 苹果"}, [(3, "苹果", "E", 4), (5, "苹果", "E", 4)])

    routes = await route(
        store, atom_ids={("苹果", "E"): apple}, plan=plan, context_vectors={"妈妈 买 苹果": list(basis(0))}
    )

    assert set(routes) == {(apple, 4)}  # one deduplicated routing entry
    rows = store.anchors_rows()
    assert len(rows) == 1
    assert rows[0]["total_count"] == 1  # sampled once, not once per occurrence


async def test_same_atom_different_frames_route_independently():
    store = FakeStore()
    apple = seed_atom(store, "苹果", "E")
    plan = make_plan(
        {2: "妈妈 买 苹果", 4: "小明 吃 苹果"},
        [(3, "苹果", "E", 2), (5, "苹果", "E", 4)],
    )

    routes = await route(
        store,
        atom_ids={("苹果", "E"): apple},
        plan=plan,
        context_vectors={"妈妈 买 苹果": list(basis(0)), "小明 吃 苹果": list(basis(1))},
    )

    assert set(routes) == {(apple, 2), (apple, 4)}
    assert routes[(apple, 2)] != routes[(apple, 4)]  # orthogonal frames → separate anchors
    rows = store.anchors_rows()
    assert len(rows) == 2
    assert all(row["total_count"] == 1 for row in rows)


async def test_anchor_sql_never_reads_support_count():
    store = FakeStore()
    atom_id = seed_atom(store, "苹果", "E")
    plan = make_plan({4: "妈妈 买 苹果"}, [(5, "苹果", "E", 4)])

    await route(store, atom_ids={("苹果", "E"): atom_id}, plan=plan, context_vectors={"妈妈 买 苹果": list(basis(0))})

    calls = anchor_sql_calls(store)
    assert calls  # anchors SQL did run
    for sql in calls:
        assert "support_count" not in sql


async def test_route_never_writes_null_anchor():
    store = FakeStore()
    apple = seed_atom(store, "苹果", "E")
    plan = make_plan({4: "妈妈 买 苹果"}, [(5, "苹果", "E", 4)])

    routes = await route(
        store, atom_ids={("苹果", "E"): apple}, plan=plan, context_vectors={"妈妈 买 苹果": list(basis(0))}
    )

    assert routes
    assert set(routes) == {(apple, 4)}  # every (atom_id, frame_pos) covered
    assert all(anchor_id is not None for anchor_id in routes.values())


# ---------------------------------------------------------------------------
# Context Vector preparation (requirement 05 §7.5 / §8.1)
# ---------------------------------------------------------------------------


async def test_prepare_context_vectors_embeds_each_unique_text_once():
    client = FakeEmbeddingClient()
    plan = AnchorPlan(
        frames={2: PredicateFrame(2, "妈妈 让 小明"), 4: PredicateFrame(4, "打 酱油")},
        occurrences=(
            OccurrenceRoute(1, ("妈妈", "E"), 2),
            OccurrenceRoute(2, ("让", "P"), 2),
            OccurrenceRoute(3, ("小明", "E"), 2),
            OccurrenceRoute(4, ("打", "P"), 4),
            OccurrenceRoute(5, ("酱油", "E"), 4),
        ),
        context_texts=("妈妈 让 小明", "打 酱油"),
    )

    vectors = await prepare_context_vectors(client, plan)

    assert client.ensure_calls == 1
    assert client.context_calls == ["妈妈 让 小明", "打 酱油"]  # plan order, no duplicates
    assert vectors == {
        "妈妈 让 小明": list(fake_context_vector("妈妈 让 小明")),
        "打 酱油": list(fake_context_vector("打 酱油")),
    }


async def test_prepare_context_vectors_failure_propagates():
    client = FakeEmbeddingClient(context_failures={"打 酱油": "http_422"})
    plan = AnchorPlan(
        frames={4: PredicateFrame(4, "打 酱油")},
        occurrences=(OccurrenceRoute(4, ("打", "P"), 4),),
        context_texts=("打 酱油",),
    )

    with pytest.raises(EmbeddingError):  # the caller maps this to anchor_context_* alerts
        await prepare_context_vectors(client, plan)
