from __future__ import annotations

import math
from pathlib import Path

import pytest

from hindsight_api.engine.retain.noesis_anchor_compaction import (
    AnchorSnapshot,
    DottedUnavailableError,
    PairEvidence,
    apply_merge,
    choose_survivor,
    combine_dotted_scores,
    jaccard,
    merge_centroids,
    plan_atom_compaction,
    rank_pairs,
    role_aware_jaccard,
)


def anchor(anchor_id: int, count: int, value: float = 0.0) -> AnchorSnapshot:
    return AnchorSnapshot(
        anchor_id=anchor_id,
        atom_id=7,
        total_count=count,
        centroid=(value,) * 1024,
        updated_at="v1",
    )


def test_merge_centroids_uses_support_weighted_mean():
    merged = merge_centroids(anchor(1, 3, 0.0), anchor(2, 1, 1.0))
    assert merged == pytest.approx((0.25,) * 1024)


def test_choose_survivor_prefers_count_then_lower_id():
    assert choose_survivor(anchor(9, 4), anchor(3, 2))[0].anchor_id == 9
    assert choose_survivor(anchor(9, 4), anchor(3, 4))[0].anchor_id == 3


def test_jaccard_and_role_aware_predicate_average_ignore_double_empty():
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    # S is informative; O is empty on both sides and must not dilute the score.
    assert role_aware_jaccard("P", {"S": {1, 2}, "O": set()}, {"S": {2, 3}, "O": set()}) == pytest.approx(1 / 3)
    assert role_aware_jaccard("E", {"N": {1, 2}}, {"N": {2}}) == pytest.approx(0.5)


def test_dotted_score_is_symmetric_median():
    assert combine_dotted_scores([0.9, 0.8, 0.1, 0.2]) == pytest.approx(0.5)
    with pytest.raises(DottedUnavailableError):
        combine_dotted_scores([])


def test_pair_ranking_uses_dotted_then_jaccard_distance_and_ids():
    pairs = [
        PairEvidence(4, 5, 0.1, 0.9, 0.7, False),
        PairEvidence(2, 3, 0.2, 0.8, 0.8, False),
        PairEvidence(1, 9, 0.1, 0.8, 0.8, False),
        PairEvidence(1, 8, 0.1, 0.8, 0.8, False),
    ]
    ranked = rank_pairs(pairs)
    assert [(p.left_anchor_id, p.right_anchor_id) for p in ranked] == [
        (1, 8),
        (1, 9),
        (2, 3),
        (4, 5),
    ]


def test_non_finite_pair_evidence_is_rejected():
    with pytest.raises(ValueError):
        PairEvidence(1, 2, math.nan, 0.1, 0.2, False)


def test_plan_compacts_to_k_and_recomputes_after_each_merge():
    anchors = [anchor(1, 10, 0.0), anchor(2, 1, 0.01), anchor(3, 8, 0.8)]
    neighbors = {1: {"N": {10}}, 2: {"N": {10, 11}}, 3: {"N": {99}}}
    exemplars = {1: ["吃苹果"], 2: ["削苹果"], 3: ["给苹果充电"]}

    def score(_word, left, right):
        return 0.9 if "吃苹果" in left + right and "削苹果" in left + right else 0.1

    plans = plan_atom_compaction(
        atom_type="E",
        atom_text="苹果",
        anchors=anchors,
        neighbors=neighbors,
        exemplars=exemplars,
        max_active=2,
        dotted_scorer=score,
    )

    assert len(plans) == 1
    assert (plans[0].source.anchor_id, plans[0].target.anchor_id) == (2, 1)
    assert plans[0].target_after.total_count == 11


def test_candidate_dotted_calls_are_batched_once_per_merge_round():
    class Scorer:
        def __init__(self):
            self.batch_sizes = []

        def score_many(self, _word, pairs):
            self.batch_sizes.append(len(pairs))
            return [0.5] * len(pairs)

    scorer = Scorer()
    plan_atom_compaction(
        atom_type="E",
        atom_text="苹果",
        anchors=[anchor(1, 3, 0.0), anchor(2, 2, 0.1), anchor(3, 1, 0.2)],
        neighbors={1: {}, 2: {}, 3: {}},
        exemplars={1: ["吃苹果"], 2: ["削苹果"], 3: ["买苹果"]},
        max_active=2,
        dotted_scorer=scorer,
    )
    assert scorer.batch_sizes == [3]


def test_missing_exemplar_uses_forced_geometry_fallback():
    plans = plan_atom_compaction(
        atom_type="E",
        atom_text="苹果",
        anchors=[anchor(1, 2, 0.0), anchor(2, 1, 0.01)],
        neighbors={1: {"N": {10}}, 2: {"N": {10}}},
        exemplars={1: ["吃苹果"], 2: []},
        max_active=1,
        dotted_scorer=lambda *_: 1.0,
    )
    assert plans[0].evidence.forced is True


def test_dotted_total_failure_is_fail_closed():
    def unavailable(*_args):
        raise DottedUnavailableError("model unavailable")

    with pytest.raises(DottedUnavailableError):
        plan_atom_compaction(
            atom_type="E",
            atom_text="苹果",
            anchors=[anchor(1, 2), anchor(2, 1, 0.01)],
            neighbors={1: {}, 2: {}},
            exemplars={1: ["吃苹果"], 2: ["削苹果"]},
            max_active=1,
            dotted_scorer=unavailable,
        )


@pytest.mark.asyncio
async def test_apply_merge_only_writes_owned_fact_metric_anchor_and_audit_tables():
    class Conn:
        def __init__(self):
            self.calls = []

        async def fetchval(self, sql, *args):
            self.calls.append(("fetchval", sql, args))
            if ".event_atoms" in sql:
                return 3
            if ".cooccurrence_bitmaps" in sql:
                return 1
            if ".neighbor_bitmaps" in sql:
                return 2
            raise AssertionError(sql)

        async def execute(self, sql, *args):
            self.calls.append(("execute", sql, args))

    conn = Conn()
    target = anchor(1, 4, 0.0)
    source = anchor(2, 2, 1.0)
    evidence = PairEvidence(1, 2, 0.1, 0.5, 0.8, False)
    await apply_merge(
        conn,
        "noesis_anchor_lab",
        run_id=9,
        atom_id=7,
        target=target,
        source=source,
        evidence=evidence,
        downstream_impact={"class_membership": 2, "rules": 0},
    )

    sql = "\n".join(call[1] for call in conn.calls)
    assert "UPDATE noesis_anchor_lab.event_atoms SET anchor_id=$3" in sql
    assert sql.count("rb64_or") == 2
    assert "SET status='M'" in sql
    assert "anchor_merge_items" in sql
    assert not any(
        token in sql
        for token in (
            "UPDATE noesis_anchor_lab.class_",
            "DELETE FROM noesis_anchor_lab.class_",
            "UPDATE noesis_anchor_lab.rules",
            "UPDATE noesis_anchor_lab.cognitive_assertions",
        )
    )
    audit = next(call for call in conn.calls if "anchor_merge_items" in call[1])
    assert audit[2][11:14] == (3, 1, 2)


def test_migration_010_declares_append_only_merge_audit_contract():
    root = Path(__file__).resolve().parents[3]
    migration = (root / "docs/db/migrations/010-noesis-anchor-compaction-audit.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS noesis_core.anchor_merge_runs" in migration
    assert "CREATE TABLE IF NOT EXISTS noesis_core.anchor_merge_items" in migration
    assert "UNIQUE (source_anchor_id)" in migration
    assert "source_anchor_id <> target_anchor_id" in migration
    forbidden = ("UPDATE noesis_core.anchors", "DELETE FROM noesis_core.")
    assert not any(token in migration for token in forbidden)
