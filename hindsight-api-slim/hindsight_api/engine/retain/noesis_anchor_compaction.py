"""Bounded asynchronous consolidation for Noesis Anchors.

The online router may create one temporary ``K+1`` anchor.  This module plans
and applies a per-Atom merge back to ``K`` while leaving a complete audit
record.  Semantic scoring happens before the transaction; the transaction
revalidates the exact anchor snapshot and performs only deterministic remap,
bitmap union, anchor state, and audit writes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import asyncpg

from hindsight_api.db_url import to_libpq_url
from hindsight_api.engine.retain.noesis_anchor import parse_centroid_text, vector_literal

ALGORITHM_VERSION = "anchor-cap-v1"


class AnchorCompactionError(Exception):
    """A compaction run cannot safely continue."""


class DottedUnavailableError(AnchorCompactionError):
    """The configured Dotted-WSD scorer is unavailable; fail closed."""


class DottedExemplarError(AnchorCompactionError):
    """One pair lacks a usable unambiguous historical occurrence."""


class StaleCompactionPlan(AnchorCompactionError):
    """An Atom changed after scoring; no writes were applied for it."""


@dataclass(frozen=True)
class AnchorSnapshot:
    anchor_id: int
    atom_id: int
    total_count: int
    centroid: tuple[float, ...]
    updated_at: Any

    def __post_init__(self) -> None:
        if self.anchor_id <= 0 or self.atom_id <= 0 or self.total_count <= 0:
            raise ValueError("anchor ids and total_count must be positive")
        if len(self.centroid) != 1024 or not all(math.isfinite(value) for value in self.centroid):
            raise ValueError("centroid must contain exactly 1024 finite values")


@dataclass(frozen=True)
class PairEvidence:
    left_anchor_id: int
    right_anchor_id: int
    centroid_distance: float
    neighbor_jaccard: float
    dotted_score: float
    forced: bool

    def __post_init__(self) -> None:
        if self.left_anchor_id == self.right_anchor_id:
            raise ValueError("pair requires two different anchors")
        values = (self.centroid_distance, self.neighbor_jaccard, self.dotted_score)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pair evidence must be finite")


@dataclass(frozen=True)
class MergePlan:
    source: AnchorSnapshot
    target: AnchorSnapshot
    target_after: AnchorSnapshot
    evidence: PairEvidence


def merge_centroids(left: AnchorSnapshot, right: AnchorSnapshot) -> tuple[float, ...]:
    """Return the exact support-weighted centroid for two anchors."""
    total = left.total_count + right.total_count
    return tuple((left.total_count * a + right.total_count * b) / total for a, b in zip(left.centroid, right.centroid))


def choose_survivor(left: AnchorSnapshot, right: AnchorSnapshot) -> tuple[AnchorSnapshot, AnchorSnapshot]:
    """Choose target by support, with lower id as the deterministic tie-break."""
    ordered = sorted((left, right), key=lambda item: (-item.total_count, item.anchor_id))
    return ordered[0], ordered[1]


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def role_aware_jaccard(
    atom_type: str,
    left: Mapping[str, set[int]],
    right: Mapping[str, set[int]],
) -> float:
    """Entity uses N; predicate averages non-double-empty S/O roles."""
    roles = ("N",) if atom_type == "E" else ("S", "O")
    scores = [
        jaccard(left.get(role, set()), right.get(role, set()))
        for role in roles
        if left.get(role, set()) or right.get(role, set())
    ]
    return statistics.fmean(scores) if scores else 0.0


def combine_dotted_scores(scores: Sequence[float]) -> float:
    """Aggregate symmetric AB/BA exemplar-pair probabilities by median."""
    if not scores:
        raise DottedUnavailableError("no Dotted-WSD scores were produced")
    if not all(math.isfinite(score) for score in scores):
        raise DottedUnavailableError("Dotted-WSD returned a non-finite score")
    return float(statistics.median(scores))


def rank_pairs(pairs: Iterable[PairEvidence]) -> list[PairEvidence]:
    return sorted(
        pairs,
        key=lambda pair: (
            -pair.dotted_score,
            -pair.neighbor_jaccard,
            pair.centroid_distance,
            min(pair.left_anchor_id, pair.right_anchor_id),
            max(pair.left_anchor_id, pair.right_anchor_id),
        ),
    )


def _mark_single_occurrence(sentence: str, word: str) -> str | None:
    if not word or sentence.count(word) != 1:
        return None
    return sentence.replace(word, f"<{word}>", 1)


class DottedWSDScorer:
    """Lazy, offline-only Dotted-WSD pair scorer used by the background CLI."""

    def __init__(self, model_dir: str | Path, *, device: str = "cpu") -> None:
        self._model_dir = Path(model_dir)
        self._device = device
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._yes_index: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_dir, local_files_only=True).to(
                self._device
            )
            self._model.eval()
            self._torch = torch
            labels = getattr(self._model.config, "id2label", {}) or {}
            self._yes_index = next(
                (
                    int(index)
                    for index, label in labels.items()
                    if str(label).upper() in {"YES", "LABEL_1", "ENTAILMENT", "TRUE"}
                ),
                1,
            )
        except Exception as error:
            raise DottedUnavailableError(f"Dotted-WSD load failed: {type(error).__name__}") from error

    def __call__(self, word: str, left: Sequence[str], right: Sequence[str]) -> float:
        return self.score_many(word, [(left, right)])[0]

    def score_many(self, word: str, pairs: Sequence[tuple[Sequence[str], Sequence[str]]]) -> list[float]:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._yes_index is not None
        contexts: list[str] = []
        candidates: list[str] = []
        owners: list[int] = []
        for owner, (left, right) in enumerate(pairs):
            left_marked = [marked for value in left if (marked := _mark_single_occurrence(value, word))]
            right_marked = [marked for value in right if (marked := _mark_single_occurrence(value, word))]
            if not left_marked or not right_marked:
                raise DottedExemplarError("no unambiguous Dotted-WSD exemplar occurrence")
            left_gloss = f"{word},该义项曾出现在以下语境：{'；'.join(left)}"
            right_gloss = f"{word},该义项曾出现在以下语境：{'；'.join(right)}"
            contexts.extend(left_marked + right_marked)
            candidates.extend([right_gloss] * len(left_marked) + [left_gloss] * len(right_marked))
            owners.extend([owner] * (len(left_marked) + len(right_marked)))
        try:
            grouped: list[list[float]] = [[] for _ in pairs]
            for start in range(0, len(contexts), 16):
                encoded = self._tokenizer(
                    contexts[start : start + 16],
                    candidates[start : start + 16],
                    padding=True,
                    truncation=True,
                    max_length=320,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                with self._torch.inference_mode():
                    logits = self._model(**encoded).logits.float()
                    probabilities = self._torch.softmax(logits, dim=-1)[:, self._yes_index]
                for owner, value in zip(owners[start : start + 16], probabilities.cpu().tolist(), strict=True):
                    grouped[owner].append(float(value))
            return [combine_dotted_scores(scores) for scores in grouped]
        except DottedUnavailableError:
            raise
        except Exception as error:
            raise DottedUnavailableError(f"Dotted-WSD inference failed: {type(error).__name__}") from error


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    return 1.0 - dot / (left_norm * right_norm)


def candidate_pairs(
    anchors: Sequence[AnchorSnapshot],
    jaccards: Mapping[tuple[int, int], float],
    *,
    limit: int = 8,
) -> set[tuple[int, int]]:
    """Union the nearest-centroid and highest-Jaccard bounded candidate sets."""
    distances: list[tuple[float, int, int]] = []
    for index, left in enumerate(anchors):
        for right in anchors[index + 1 :]:
            a, b = sorted((left.anchor_id, right.anchor_id))
            distances.append((cosine_distance(left.centroid, right.centroid), a, b))
    by_distance = {(a, b) for _, a, b in sorted(distances)[:limit]}
    by_jaccard = {pair for pair, _ in sorted(jaccards.items(), key=lambda item: (-item[1], item[0]))[:limit]}
    return by_distance | by_jaccard


def _bounded_exemplars(values: Sequence[str]) -> list[str]:
    ordered = list(dict.fromkeys(value for value in values if value))
    if len(ordered) <= 3:
        return ordered
    return [ordered[0], ordered[len(ordered) // 2], ordered[-1]]


def plan_atom_compaction(
    *,
    atom_type: str,
    atom_text: str,
    anchors: Sequence[AnchorSnapshot],
    neighbors: Mapping[int, Mapping[str, set[int]]],
    exemplars: Mapping[int, Sequence[str]],
    max_active: int,
    dotted_scorer: Any,
) -> list[MergePlan]:
    """Plan all merges for one Atom without holding a database lock.

    The in-memory state is updated after each selected pair, so a historical
    Atom with many anchors is not planned as independent, conflicting merges.
    A missing exemplar on an individual pair enables the documented forced
    Jaccard/centroid fallback.  A scorer exception is not a pair fallback: it
    aborts the whole Atom fail-closed.
    """
    if max_active < 1:
        raise ValueError("max_active must be positive")
    current = {item.anchor_id: item for item in anchors}
    current_neighbors = {
        anchor_id: {role: set(bits) for role, bits in role_map.items()} for anchor_id, role_map in neighbors.items()
    }
    current_exemplars = {anchor_id: _bounded_exemplars(values) for anchor_id, values in exemplars.items()}
    plans: list[MergePlan] = []
    while len(current) > max_active:
        ordered = sorted(current.values(), key=lambda item: item.anchor_id)
        jaccards: dict[tuple[int, int], float] = {}
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                pair = (left.anchor_id, right.anchor_id)
                jaccards[pair] = role_aware_jaccard(
                    atom_type,
                    current_neighbors.get(left.anchor_id, {}),
                    current_neighbors.get(right.anchor_id, {}),
                )
        pending: list[tuple[int, int, list[str], list[str], bool]] = []
        for left_id, right_id in sorted(candidate_pairs(ordered, jaccards)):
            left = current[left_id]
            right = current[right_id]
            left_examples = [
                value
                for value in current_exemplars.get(left_id, [])
                if _mark_single_occurrence(value, atom_text) is not None
            ]
            right_examples = [
                value
                for value in current_exemplars.get(right_id, [])
                if _mark_single_occurrence(value, atom_text) is not None
            ]
            forced = dotted_scorer is None or not left_examples or not right_examples
            pending.append((left_id, right_id, left_examples, right_examples, forced))
        scoreable = [(left, right) for _, _, left, right, forced in pending if not forced]
        if dotted_scorer is None:
            scored = iter(())
        elif hasattr(dotted_scorer, "score_many"):
            scored = iter(dotted_scorer.score_many(atom_text, scoreable))
        else:
            scored = iter(dotted_scorer(atom_text, left, right) for left, right in scoreable)
        evidence: list[PairEvidence] = []
        for left_id, right_id, _left_examples, _right_examples, forced in pending:
            dotted = 0.0 if forced else float(next(scored))
            evidence.append(
                PairEvidence(
                    left_id,
                    right_id,
                    cosine_distance(left.centroid, right.centroid),
                    jaccards[(left_id, right_id)],
                    dotted,
                    forced,
                )
            )
        if not evidence:
            raise AnchorCompactionError("no merge candidate could be constructed")
        selected = rank_pairs(evidence)[0]
        target, source = choose_survivor(current[selected.left_anchor_id], current[selected.right_anchor_id])
        target_after = AnchorSnapshot(
            anchor_id=target.anchor_id,
            atom_id=target.atom_id,
            total_count=target.total_count + source.total_count,
            centroid=merge_centroids(target, source),
            updated_at=target.updated_at,
        )
        plans.append(MergePlan(source, target, target_after, selected))
        current[target.anchor_id] = target_after
        del current[source.anchor_id]
        merged_roles: dict[str, set[int]] = {}
        for role in {"N", "S", "O"}:
            merged_roles[role] = current_neighbors.get(target.anchor_id, {}).get(role, set()) | current_neighbors.get(
                source.anchor_id, {}
            ).get(role, set())
        current_neighbors[target.anchor_id] = merged_roles
        current_neighbors.pop(source.anchor_id, None)
        current_exemplars[target.anchor_id] = _bounded_exemplars(
            current_exemplars.get(target.anchor_id, []) + current_exemplars.get(source.anchor_id, [])
        )
        current_exemplars.pop(source.anchor_id, None)
    return plans


def _sql(schema: str, statement: str) -> str:
    if not schema or not schema.replace("_", "a").isalnum() or schema[0].isdigit():
        raise ValueError(f"invalid schema name {schema!r}")
    return statement.format(s=schema)


_ACTIVE_ANCHORS = (
    "SELECT anchor_id, atom_id, total_count, centroid_vector::text AS centroid_text, updated_at "
    "FROM {s}.anchors WHERE atom_id=$1 AND status='A' ORDER BY anchor_id"
)


async def load_anchor_snapshots(conn: Any, schema: str, atom_id: int) -> list[AnchorSnapshot]:
    rows = await conn.fetch(_sql(schema, _ACTIVE_ANCHORS), atom_id)
    return [
        AnchorSnapshot(
            anchor_id=int(row["anchor_id"]),
            atom_id=int(row["atom_id"]),
            total_count=int(row["total_count"]),
            centroid=tuple(parse_centroid_text(row["centroid_text"])),
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


async def apply_merge(
    conn: Any,
    schema: str,
    *,
    run_id: int,
    atom_id: int,
    target: AnchorSnapshot,
    source: AnchorSnapshot,
    evidence: PairEvidence,
    downstream_impact: Mapping[str, int],
) -> None:
    """Apply one already-revalidated merge inside the caller transaction.

    ``event_atoms`` is remapped in place. Its frozen primary key is
    ``(event_id, occurrence_id)``, so two physical occurrences remain two rows
    even when consolidation gives them the same Anchor.
    """
    merged = merge_centroids(target, source)
    moved = await conn.fetchval(
        _sql(
            schema,
            "WITH updated AS (UPDATE {s}.event_atoms SET anchor_id=$3 WHERE atom_id=$1 AND anchor_id=$2 RETURNING 1) "
            "SELECT (SELECT count(*) FROM updated)",
        ),
        atom_id,
        source.anchor_id,
        target.anchor_id,
    )
    co_rows = await conn.fetchval(
        _sql(
            schema,
            "WITH src AS (DELETE FROM {s}.cooccurrence_bitmaps WHERE atom_id=$1 AND anchor_id=$2 "
            "RETURNING event_bitmap), ins AS (INSERT INTO {s}.cooccurrence_bitmaps(atom_id,anchor_id,event_bitmap) "
            "SELECT $1,$3,event_bitmap FROM src ON CONFLICT(atom_id,anchor_id) DO UPDATE "
            "SET event_bitmap=rb64_or({s}.cooccurrence_bitmaps.event_bitmap,EXCLUDED.event_bitmap) RETURNING 1) "
            "SELECT count(*) FROM ins",
        ),
        atom_id,
        source.anchor_id,
        target.anchor_id,
    )
    neighbor_rows = await conn.fetchval(
        _sql(
            schema,
            "WITH src AS (DELETE FROM {s}.neighbor_bitmaps WHERE atom_id=$1 AND anchor_id=$2 "
            "RETURNING role_type,neighbor_bitmap), ins AS (INSERT INTO {s}.neighbor_bitmaps(atom_id,anchor_id,role_type,neighbor_bitmap) "
            "SELECT $1,$3,role_type,neighbor_bitmap FROM src ON CONFLICT(atom_id,anchor_id,role_type) DO UPDATE "
            "SET neighbor_bitmap=rb64_or({s}.neighbor_bitmaps.neighbor_bitmap,EXCLUDED.neighbor_bitmap) RETURNING 1) "
            "SELECT count(*) FROM ins",
        ),
        atom_id,
        source.anchor_id,
        target.anchor_id,
    )

    await conn.execute(
        _sql(
            schema,
            "UPDATE {s}.anchors SET centroid_vector=$2::vector,total_count=$3,updated_at=now() "
            "WHERE anchor_id=$1 AND status='A'",
        ),
        target.anchor_id,
        vector_literal(merged),
        target.total_count + source.total_count,
    )
    await conn.execute(
        _sql(schema, "UPDATE {s}.anchors SET status='M',updated_at=now() WHERE anchor_id=$1 AND status='A'"),
        source.anchor_id,
    )
    await conn.execute(
        _sql(
            schema,
            "INSERT INTO {s}.anchor_merge_items(run_id,atom_id,source_anchor_id,target_anchor_id,merged_at,"
            "source_total_count,target_total_count_before,target_total_count_after,centroid_distance,"
            "neighbor_jaccard,dotted_score,forced,event_atoms_moved,cooccurrence_rows_merged,"
            "neighbor_rows_merged,downstream_impact,target_centroid_before,target_centroid_after) "
            "VALUES($1,$2,$3,$4,now(),$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15::jsonb,$16::vector,$17::vector)",
        ),
        run_id,
        atom_id,
        source.anchor_id,
        target.anchor_id,
        source.total_count,
        target.total_count,
        target.total_count + source.total_count,
        evidence.centroid_distance,
        evidence.neighbor_jaccard,
        evidence.dotted_score,
        evidence.forced,
        int(moved or 0),
        int(co_rows or 0),
        int(neighbor_rows or 0),
        json.dumps(dict(downstream_impact), sort_keys=True),
        vector_literal(target.centroid),
        vector_literal(merged),
    )


async def _load_atom_inputs(
    pool: Any, schema: str, atom_id: int
) -> tuple[str, str, list[AnchorSnapshot], dict[int, dict[str, set[int]]], dict[int, list[str]]]:
    async with pool.acquire() as conn:
        atom = await conn.fetchrow(
            _sql(schema, "SELECT text,atom_type::text AS atom_type FROM {s}.atoms WHERE atom_id=$1"),
            atom_id,
        )
        if atom is None:
            raise AnchorCompactionError(f"atom {atom_id} disappeared")
        anchors = await load_anchor_snapshots(conn, schema, atom_id)
        neighbor_rows = await conn.fetch(
            _sql(
                schema,
                "SELECT anchor_id,role_type::text AS role_type,rb64_to_array(neighbor_bitmap) AS ids "
                "FROM {s}.neighbor_bitmaps WHERE atom_id=$1",
            ),
            atom_id,
        )
        exemplar_rows = await conn.fetch(
            _sql(
                schema,
                "WITH ranked AS (SELECT ea.anchor_id,e.data->>'source_text' AS source_text,"
                "row_number() OVER(PARTITION BY ea.anchor_id ORDER BY e.event_time,e.event_id) AS rn,"
                "count(*) OVER(PARTITION BY ea.anchor_id) AS cnt FROM {s}.event_atoms ea "
                "JOIN {s}.events e ON e.event_id=ea.event_id WHERE ea.atom_id=$1) "
                "SELECT anchor_id,source_text FROM ranked WHERE rn=1 OR rn=(cnt+1)/2 OR rn=cnt "
                "ORDER BY anchor_id,rn",
            ),
            atom_id,
        )
    neighbors: dict[int, dict[str, set[int]]] = {}
    for row in neighbor_rows:
        neighbors.setdefault(int(row["anchor_id"]), {})[row["role_type"]] = set(row["ids"] or [])
    exemplars: dict[int, list[str]] = {}
    for row in exemplar_rows:
        if row["source_text"]:
            exemplars.setdefault(int(row["anchor_id"]), []).append(row["source_text"])
    return atom["atom_type"], atom["text"], anchors, neighbors, exemplars


def _snapshot_key(anchor: AnchorSnapshot) -> tuple[Any, ...]:
    return (
        anchor.anchor_id,
        anchor.atom_id,
        anchor.total_count,
        anchor.centroid,
        anchor.updated_at,
    )


async def _downstream_impact(conn: Any, schema: str, anchor_id: int) -> dict[str, int]:
    tables = (
        "class_membership",
        "class_bitmaps",
        "classes",
        "rules",
        "rule_premises",
        "cognitive_assertions",
    )
    impact: dict[str, int] = {}
    for table in tables:
        exists = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", f"{schema}.{table}")
        if not exists:
            impact[table] = 0
            continue
        if table == "class_membership":
            impact[table] = int(
                await conn.fetchval(
                    _sql(schema, "SELECT count(*) FROM {s}.class_membership WHERE anchor_id=$1"),
                    anchor_id,
                )
            )
        else:
            # These downstream tables have no direct Anchor FK in the frozen
            # schema. Record zero; never infer or mutate indirect ownership.
            impact[table] = 0
    return impact


async def run_compaction(
    pool: Any,
    *,
    schema: str,
    max_active: int = 5,
    max_atoms: int = 5,
    dry_run: bool = False,
    atom_id: int | None = None,
    run_id: int | None = None,
    dotted_scorer: Any,
) -> dict[str, int | str]:
    """Run one bounded scan.  Each Atom's planned merges commit atomically."""
    if max_active < 1 or max_atoms < 1:
        raise ValueError("max_active and max_atoms must be positive")
    config = {"max_active": max_active, "max_atoms": max_atoms, "algorithm": ALGORITHM_VERSION}
    async with pool.acquire() as conn:
        if run_id is None:
            run_id = int(
                await conn.fetchval(
                    _sql(
                        schema,
                        "INSERT INTO {s}.anchor_merge_runs(schema_name,mode,status,algorithm_version,max_active,config_snapshot) "
                        "VALUES($1,$2,'running',$3,$4,$5::jsonb) RETURNING run_id",
                    ),
                    schema,
                    "dry_run" if dry_run else "apply",
                    ALGORITHM_VERSION,
                    max_active,
                    json.dumps(config, sort_keys=True),
                )
            )
        candidates = (
            await conn.fetch(
                _sql(
                    schema,
                    "SELECT atom_id FROM {s}.anchors WHERE status='A' "
                    + ("AND atom_id=$2 " if atom_id is not None else "")
                    + "GROUP BY atom_id HAVING count(*)>$1 ORDER BY count(*) DESC,atom_id LIMIT $3",
                ),
                max_active,
                atom_id,
                max_atoms,
            )
            if atom_id is not None
            else await conn.fetch(
                _sql(
                    schema,
                    "SELECT atom_id FROM {s}.anchors WHERE status='A' GROUP BY atom_id "
                    "HAVING count(*)>$1 ORDER BY count(*) DESC,atom_id LIMIT $2",
                ),
                max_active,
                max_atoms,
            )
        )
        await conn.execute(
            _sql(schema, "UPDATE {s}.anchor_merge_runs SET candidate_atoms=$2 WHERE run_id=$1"),
            run_id,
            len(candidates),
        )
    merged_pairs = 0
    stale = False
    planned: list[dict[str, Any]] = []
    try:
        for candidate in candidates:
            current_atom_id = int(candidate["atom_id"])
            atom_type, atom_text, anchors, neighbors, exemplars = await _load_atom_inputs(pool, schema, current_atom_id)
            plans = plan_atom_compaction(
                atom_type=atom_type,
                atom_text=atom_text,
                anchors=anchors,
                neighbors=neighbors,
                exemplars=exemplars,
                max_active=max_active,
                dotted_scorer=dotted_scorer,
            )
            if dry_run:
                planned.extend(
                    {
                        "atom_id": current_atom_id,
                        "source_anchor_id": plan.source.anchor_id,
                        "target_anchor_id": plan.target.anchor_id,
                        "forced": plan.evidence.forced,
                        "dotted_score": round(plan.evidence.dotted_score, 6),
                        "neighbor_jaccard": round(plan.evidence.neighbor_jaccard, 6),
                        "centroid_distance": round(plan.evidence.centroid_distance, 6),
                    }
                    for plan in plans
                )
                merged_pairs += len(plans)
                continue
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.fetchrow(
                        _sql(schema, "SELECT atom_id FROM {s}.atoms WHERE atom_id=$1 FOR UPDATE"),
                        current_atom_id,
                    )
                    locked_rows = await conn.fetch(_sql(schema, _ACTIVE_ANCHORS + " FOR UPDATE"), current_atom_id)
                    locked = [
                        AnchorSnapshot(
                            int(row["anchor_id"]),
                            int(row["atom_id"]),
                            int(row["total_count"]),
                            tuple(parse_centroid_text(row["centroid_text"])),
                            row["updated_at"],
                        )
                        for row in locked_rows
                    ]
                    if [_snapshot_key(value) for value in locked] != [_snapshot_key(value) for value in anchors]:
                        stale = True
                        raise StaleCompactionPlan(f"atom {current_atom_id} changed during scoring")
                    for plan in plans:
                        impact = await _downstream_impact(conn, schema, plan.source.anchor_id)
                        await apply_merge(
                            conn,
                            schema,
                            run_id=run_id,
                            atom_id=current_atom_id,
                            target=plan.target,
                            source=plan.source,
                            evidence=plan.evidence,
                            downstream_impact=impact,
                        )
                    merged_pairs += len(plans)
        status = "stale" if stale else "completed"
        async with pool.acquire() as conn:
            await conn.execute(
                _sql(
                    schema,
                    "UPDATE {s}.anchor_merge_runs SET status=$2,merged_pairs=$3,finished_at=now() WHERE run_id=$1",
                ),
                run_id,
                status,
                merged_pairs,
            )
        result = {
            "run_id": run_id,
            "status": status,
            "candidate_atoms": len(candidates),
            "merged_pairs": merged_pairs,
        }
        if dry_run:
            result["planned"] = planned
        return result
    except BaseException as error:
        async with pool.acquire() as conn:
            await conn.execute(
                _sql(
                    schema,
                    "UPDATE {s}.anchor_merge_runs SET status=$2,merged_pairs=$3,finished_at=now(),error_code=$4 WHERE run_id=$1",
                ),
                run_id,
                "stale" if isinstance(error, StaleCompactionPlan) else "failed",
                merged_pairs,
                type(error).__name__,
            )
        if isinstance(error, StaleCompactionPlan):
            return {
                "run_id": run_id,
                "status": "stale",
                "candidate_atoms": len(candidates),
                "merged_pairs": merged_pairs,
            }
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="noesis-anchor-compaction")
    parser.add_argument("--schema", default="noesis_core")
    parser.add_argument("--max-active", type=int, default=5)
    parser.add_argument("--max-atoms", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--atom-id", type=int)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--database-url")
    parser.add_argument(
        "--dotted-model",
        default="/data/models/lopentu__google-bert-bert-base-chinese-DottedWSD",
    )
    parser.add_argument(
        "--no-dotted",
        action="store_true",
        help="control run: rank pairs by Neighbor Jaccard + centroid alone",
    )
    return parser


async def _async_main(args: argparse.Namespace) -> None:
    database_url = args.database_url or os.getenv(
        "HINDSIGHT_API_NOESIS_DATABASE_URL", "postgresql://postgres@localhost:5432/noesis"
    )
    pool = await asyncpg.create_pool(to_libpq_url(database_url), min_size=1, max_size=2)
    try:
        scorer = None if args.no_dotted else DottedWSDScorer(args.dotted_model)
        result = await run_compaction(
            pool,
            schema=args.schema,
            max_active=args.max_active,
            max_atoms=args.max_atoms,
            dry_run=args.dry_run,
            atom_id=args.atom_id,
            run_id=args.run_id,
            dotted_scorer=scorer,
        )
        print(json.dumps(result, sort_keys=True))
    finally:
        await pool.close()


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
