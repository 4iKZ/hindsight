"""Noesis Stage 1 anchor routing (requirement 05).

Focused module for the pipeline stage between the atom upsert and the
``event_atoms`` insert: predicate-frame construction and the occurrence →
frame mapping over ``target_occ`` (requirement 05 §5), Context Vector
preparation, and the in-transaction Anchor SQL routing — nearest-active reuse,
immediate create, EMA centroid update, and strict vector parsing (§6). The
public API and the docstring contracts below are frozen by
``tests/test_noesis_anchor_routing.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Cosine DISTANCE reuse gate (requirement 05 §6.4): reuse when
# ``distance <= REUSE_MAX_DISTANCE``, i.e. cosine similarity >= 0.75.
# 0.75 must NEVER be used as a distance threshold.
REUSE_MAX_DISTANCE = 0.25

# Anchor routing SQL (requirement 05 §6.3–§6.6). ``{s}`` is the schema
# placeholder filled by the local ``_sql`` helper; the statement shapes are
# frozen by the docstrings below and by the fake-store dispatch in
# ``tests/noesis_fakes.py``.
_ATOM_LOCK_SQL = "SELECT atom_id FROM {s}.atoms WHERE atom_id = $1 FOR UPDATE"
_NEAREST_ANCHOR_SQL = (
    "SELECT anchor_id, centroid_vector::text AS centroid_text, "
    "(centroid_vector <=> $2::vector) AS distance "
    "FROM {s}.anchors WHERE atom_id = $1 AND status = 'A' "
    "ORDER BY distance, anchor_id LIMIT 1"
)
_ANCHOR_EMA_UPDATE_SQL = (
    "UPDATE {s}.anchors SET centroid_vector = $2::vector, "
    "total_count = total_count + 1, updated_at = now() "
    "WHERE anchor_id = $1"
)
_ANCHOR_INSERT_SQL = (
    "INSERT INTO {s}.anchors (atom_id, centroid_vector, total_count, status) "
    "VALUES ($1, $2::vector, 1, 'A') RETURNING anchor_id"
)


def _sql(schema: str, statement: str) -> str:
    return statement.format(s=schema)


class AnchorFrameError(Exception):
    """An occurrence cannot be assigned to a predicate frame (requirement 05 §5.4)."""

    def __init__(self, message: str, *, predicate_pos: int | None = None) -> None:
        super().__init__(message)
        self.predicate_pos = predicate_pos


class AnchorRouteError(Exception):
    """Anchor SQL, vector parse, or EMA failure inside the fact transaction."""


@dataclass(frozen=True)
class PredicateFrame:
    """One predicate frame: the predicate occurrence and its context text."""

    predicate_pos: int
    context_text: str


@dataclass(frozen=True)
class OccurrenceRoute:
    """One atom occurrence pinned to the predicate frame that routes its anchor."""

    pos: int
    literal: tuple[str, str]  # (text, atom_type)
    frame_pos: int


@dataclass(frozen=True)
class AnchorPlan:
    """Frozen routing plan for one fact component."""

    frames: dict[int, PredicateFrame]  # predicate pos -> frame
    occurrences: tuple[OccurrenceRoute, ...]  # every atom occurrence, pos-ascending
    context_texts: tuple[str, ...]  # unique frame context texts, predicate-pos ascending


def _index_by_pos(atoms: Any) -> dict[int, Any]:
    """Index atoms by ``pos``; a duplicate pos means the closure contract broke."""
    by_pos: dict[int, Any] = {}
    for atom in atoms:
        if atom.pos in by_pos:
            raise AnchorFrameError(f"duplicate atom pos {atom.pos}", predicate_pos=atom.pos)
        by_pos[atom.pos] = atom
    return by_pos


def _validate_closure(by_pos: dict[int, Any]) -> None:
    """Reject broken closure invariants before framing (requirement 05 §5.1).

    A G-typed atom is not routable, an agent/patient must point at a predicate
    occurrence, the component must have exactly one root predicate, and every
    ``target_occ`` link must exist and converge to that root without
    self-points or cycles.
    """
    roots = sorted(
        (
            atom
            for atom in by_pos.values()
            if atom.role == "predicate" and atom.type == "P" and atom.target_occ is None
        ),
        key=lambda atom: atom.pos,
    )
    if len(roots) != 1:
        raise AnchorFrameError(f"component has {len(roots)} root predicates; expected exactly one")
    root_pos = roots[0].pos

    for atom in sorted(by_pos.values(), key=lambda a: a.pos):
        if atom.type == "G":
            raise AnchorFrameError(
                f"atom pos {atom.pos} has non-routable type 'G'",
                predicate_pos=atom.pos,
            )
        target = atom.target_occ
        if (
            atom.role in ("agent", "patient")
            and target is not None
            and target in by_pos
            and by_pos[target].role != "predicate"
        ):
            raise AnchorFrameError(
                f"{atom.role} pos {atom.pos} targets non-predicate pos {target}",
                predicate_pos=target,
            )
        visited = {atom.pos}
        cursor = target
        terminal_pos = atom.pos
        while cursor is not None:
            if cursor not in by_pos:
                raise AnchorFrameError(
                    f"atom pos {atom.pos} targets missing pos {cursor}",
                    predicate_pos=cursor,
                )
            if cursor in visited:
                raise AnchorFrameError(
                    f"target_occ chain from pos {atom.pos} self-points or cycles at pos {cursor}",
                    predicate_pos=cursor,
                )
            visited.add(cursor)
            terminal_pos = cursor
            cursor = by_pos[cursor].target_occ
        if terminal_pos != root_pos:
            raise AnchorFrameError(
                f"target_occ chain from pos {atom.pos} ends at pos {terminal_pos}, not root predicate pos {root_pos}",
                predicate_pos=terminal_pos,
            )


def build_predicate_frames(atoms: Any) -> dict[int, PredicateFrame]:
    """Build one frame per predicate occurrence (requirement 05 §5.2/§5.3).

    Predicates are the atoms with ``role == "predicate" and type == "P"``,
    processed in ascending ``pos`` — never in JSON array order. For each
    predicate ``p`` the agents and patients are the atoms whose
    ``target_occ == p.pos``, each sorted by ``pos``; the frame's
    ``context_text`` is ``" ".join(agents + [predicate] + patients)`` with a
    single ASCII space between elements and no role labels, punctuation,
    ``pos``, ``type``, or disambiguation suffixes. Modifiers never enter the
    context text.

    Violations raise :class:`AnchorFrameError`: an agent/patient whose
    ``target_occ`` points at a non-predicate occurrence, a G-typed atom (the
    external-G contract belongs to the upstream closure validation), a missing
    target pos, a self-pointing ``target_occ``, or a ``target_occ`` cycle.
    """
    by_pos = _index_by_pos(atoms)
    _validate_closure(by_pos)
    predicates = sorted(
        (atom for atom in by_pos.values() if atom.role == "predicate" and atom.type == "P"),
        key=lambda atom: atom.pos,
    )
    if not predicates:
        raise AnchorFrameError("component has no P predicate atom to frame")
    frames: dict[int, PredicateFrame] = {}
    for predicate in predicates:
        agents = sorted(
            (atom for atom in by_pos.values() if atom.role == "agent" and atom.target_occ == predicate.pos),
            key=lambda atom: atom.pos,
        )
        patients = sorted(
            (atom for atom in by_pos.values() if atom.role == "patient" and atom.target_occ == predicate.pos),
            key=lambda atom: atom.pos,
        )
        context_text = " ".join(
            [agent.text for agent in agents] + [predicate.text] + [patient.text for patient in patients]
        )
        frames[predicate.pos] = PredicateFrame(predicate.pos, context_text)
    return frames


def assign_occurrence_frames(atoms: Any, frames: dict[int, PredicateFrame]) -> dict[int, int]:
    """Map every atom occurrence pos to its predicate frame pos (§5.4).

    A predicate belongs to its own frame. An agent/patient must point directly
    at its predicate via ``target_occ``. A modifier walks the ``target_occ``
    chain upwards with a visited set until it reaches a predicate occurrence
    and joins that predicate's frame; a missing pos, a self-pointing target, a
    cycle, or a chain that never reaches a predicate raises
    :class:`AnchorFrameError`. A nested-clause predicate points at its parent
    layer but still belongs to its own clause frame.
    """
    by_pos = _index_by_pos(atoms)
    occurrence_frames: dict[int, int] = {}
    for atom in sorted(by_pos.values(), key=lambda a: a.pos):
        if atom.role == "predicate":
            if atom.pos not in frames:
                raise AnchorFrameError(
                    f"predicate pos {atom.pos} has no frame of its own",
                    predicate_pos=atom.pos,
                )
            occurrence_frames[atom.pos] = atom.pos
        elif atom.role in ("agent", "patient"):
            target = atom.target_occ
            if (
                target is None
                or target not in by_pos
                or by_pos[target].role != "predicate"
                or target not in frames
            ):
                raise AnchorFrameError(
                    f"{atom.role} pos {atom.pos} does not target a framed predicate",
                    predicate_pos=target,
                )
            occurrence_frames[atom.pos] = target
        else:  # modifier: walk the target_occ chain up to the nearest frame
            visited = {atom.pos}
            target = atom.target_occ
            while True:
                if target is None:
                    raise AnchorFrameError(
                        f"modifier pos {atom.pos} chain ends without reaching a predicate",
                        predicate_pos=atom.pos,
                    )
                if target not in by_pos:
                    raise AnchorFrameError(
                        f"modifier pos {atom.pos} chain hits missing pos {target}",
                        predicate_pos=target,
                    )
                if target in visited:
                    raise AnchorFrameError(
                        f"modifier pos {atom.pos} chain self-points or cycles at pos {target}",
                        predicate_pos=target,
                    )
                if target in frames:
                    occurrence_frames[atom.pos] = target
                    break
                visited.add(target)
                target = by_pos[target].target_occ
    return occurrence_frames


def plan_anchor_routing(atoms: Any) -> AnchorPlan:
    """Combine frame construction and assignment into one routing plan.

    ``occurrences`` lists every atom occurrence in ascending ``pos`` with its
    ``(text, atom_type)`` literal and its frame; ``context_texts`` lists the
    unique frame context texts in ascending predicate pos, so identical texts
    are embedded exactly once per component (requirement 05 §7.5).
    """
    atom_list = list(atoms)
    for atom in atom_list:
        if atom.type not in ("E", "P"):
            raise AnchorFrameError(
                f"atom pos {atom.pos} has non-routable type {atom.type!r}",
                predicate_pos=atom.pos,
            )
    frames = build_predicate_frames(atom_list)
    occurrence_frames = assign_occurrence_frames(atom_list, frames)
    occurrences = tuple(
        OccurrenceRoute(atom.pos, (atom.text, atom.type), occurrence_frames[atom.pos])
        for atom in sorted(atom_list, key=lambda a: a.pos)
    )
    context_texts: list[str] = []
    for predicate_pos in sorted(frames):
        context_text = frames[predicate_pos].context_text
        if context_text not in context_texts:
            context_texts.append(context_text)
    return AnchorPlan(frames=frames, occurrences=occurrences, context_texts=tuple(context_texts))


def should_reuse(distance: float) -> bool:
    """Cosine-distance reuse gate: ``distance <= REUSE_MAX_DISTANCE`` (inclusive).

    Equivalent to cosine similarity >= 0.75; 0.75 itself is never used as a
    distance threshold.
    """
    return distance <= REUSE_MAX_DISTANCE


def parse_centroid_text(text: str, dimension: int = 1024) -> list[float]:
    """Strictly parse a ``"[f1,f2,...]"`` centroid text into finite floats.

    The text must be a bracketed comma-separated list of exactly ``dimension``
    finite floats. Malformed text, a wrong dimension, or a non-finite
    component raises :class:`AnchorRouteError`. The text is never evaluated;
    the dimension is never truncated or padded; a bad vector never degrades
    into a zero vector.
    """
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise AnchorRouteError(f"malformed centroid text {stripped!r}: expected '[f1,f2,...]'")
    components = stripped[1:-1].split(",")
    if len(components) != dimension:
        raise AnchorRouteError(f"centroid dimension {len(components)} != {dimension}")
    values: list[float] = []
    for component in components:
        try:
            value = float(component)
        except ValueError as error:
            raise AnchorRouteError(f"malformed centroid component {component!r}") from error
        if not math.isfinite(value):
            raise AnchorRouteError(f"non-finite centroid component {component!r}")
        values.append(value)
    return values


def _ensure_finite_vector(vector: Any, *, what: str) -> None:
    """Reject a non-1024-dimension or non-finite vector before it reaches a write (§6.7)."""
    values = list(vector)
    if len(values) != 1024:
        raise AnchorRouteError(f"{what} dimension {len(values)} != 1024")
    for value in values:
        if not math.isfinite(value):
            raise AnchorRouteError(f"{what} contains a non-finite component")


def ema_centroid(old: list[float], context: list[float]) -> list[float]:
    """Frozen EMA: ``new[i] = 0.9 * old[i] + 0.1 * context[i]`` (§6.5).

    Both inputs must be 1024-dimension finite float lists, otherwise
    :class:`AnchorRouteError`. The result is not normalized and is not a
    cumulative average over ``total_count``.
    """
    _ensure_finite_vector(old, what="old centroid")
    _ensure_finite_vector(context, what="context vector")
    return [0.9 * old_value + 0.1 * context_value for old_value, context_value in zip(old, context)]


async def prepare_context_vectors(client: Any, plan: AnchorPlan) -> dict[str, list[float]]:
    """Embed every distinct frame context text once, outside any transaction.

    Calls ``client.ensure_ready()`` and then ``client.embed_context(text)``
    for each entry of ``plan.context_texts`` in order, each distinct text
    exactly once. Embedding failures propagate to the caller, which maps them
    to the ``anchor_context_*`` alerts; no fallback vectors are fabricated.
    """
    await client.ensure_ready()
    vectors: dict[str, list[float]] = {}
    for context_text in plan.context_texts:
        vectors[context_text] = await client.embed_context(context_text)
    return vectors


def vector_literal(vector: Any) -> str:
    """``"[f1,f2,...]"`` text literal for the ``$n::vector`` cast (§6.7).

    The same format as the identity write path: ``repr(float(...))`` keeps
    full precision and valid pgvector literal syntax. ``noesis_ingest``
    reuses this helper instead of keeping a private copy.
    """
    return "[" + ",".join(repr(float(component)) for component in vector) + "]"


async def route_anchors(
    conn: Any,
    schema: str,
    *,
    atom_ids: dict[tuple[str, str], int],
    plan: AnchorPlan,
    context_vectors: dict[str, list[float]],
) -> dict[tuple[int, int], int]:
    """Route every ``(atom_id, frame_pos)`` of the plan inside the fact transaction.

    The deduplicated set of ``(atom_ids[literal], frame_pos)`` pairs from
    ``plan.occurrences`` is processed in ascending ``atom_id`` (the lock
    order). Per entry, under the atom row lock
    (``SELECT atom_id FROM {schema}.atoms WHERE atom_id = $1 FOR UPDATE``):
    query the nearest active anchor
    (``centroid_vector <=> $2::vector``, ties broken by ``anchor_id ASC``);
    when :func:`should_reuse` accepts the distance, update it with the EMA
    centroid (``UPDATE {schema}.anchors SET centroid_vector = $2::vector,
    total_count = total_count + 1, updated_at = now() WHERE anchor_id = $1``);
    otherwise insert immediately (``INSERT INTO {schema}.anchors (atom_id,
    centroid_vector, total_count, status) VALUES ($1, $2::vector, 1, 'A')
    RETURNING anchor_id`` — the explicit ``1`` never relies on the column
    default 0). Returns ``{(atom_id, frame_pos): anchor_id}`` with no NULL
    values; the same ``(atom_id, frame_pos)`` is routed and counted exactly
    once (requirement 05 §5.6).
    """
    routes: dict[tuple[int, int], int] = {}
    try:
        entries: dict[tuple[int, int], str] = {}
        for occurrence in plan.occurrences:
            key = (atom_ids[occurrence.literal], occurrence.frame_pos)
            if key not in entries:
                entries[key] = plan.frames[occurrence.frame_pos].context_text
        for (atom_id, frame_pos), context_text in sorted(entries.items()):
            context = context_vectors[context_text]
            _ensure_finite_vector(context, what=f"context vector for atom {atom_id} frame {frame_pos}")
            literal = vector_literal(context)
            locked = await conn.fetchrow(_sql(schema, _ATOM_LOCK_SQL), atom_id)
            if locked is None:
                raise AnchorRouteError(f"atom {atom_id} row missing under FOR UPDATE lock")
            row = await conn.fetchrow(_sql(schema, _NEAREST_ANCHOR_SQL), atom_id, literal)
            if row is not None and should_reuse(row["distance"]):
                anchor_id = row["anchor_id"]
                updated = ema_centroid(parse_centroid_text(row["centroid_text"]), context)
                await conn.execute(_sql(schema, _ANCHOR_EMA_UPDATE_SQL), anchor_id, vector_literal(updated))
            else:
                inserted = await conn.fetchrow(_sql(schema, _ANCHOR_INSERT_SQL), atom_id, literal)
                anchor_id = inserted["anchor_id"]
            routes[(atom_id, frame_pos)] = anchor_id
    except AnchorRouteError:
        raise
    except Exception as error:
        raise AnchorRouteError(f"anchor routing failed: {error!r}") from error
    return routes
