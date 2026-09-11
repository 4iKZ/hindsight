"""Noesis Stage 1 bitmap write-time maintenance (requirement 06).

Focused module for the pipeline stage between the ``event_atoms`` insert and
the fact transaction COMMIT: the pure predicate-frame pairing planner and the
database-side atomic RoaringBitmap64 upserts.

Cooccurrence Bitmap — ``(atom_id, anchor_id) -> {event_id}`` for every stored
E/P occurrence (modifiers included), keyed by the real non-zero Anchor routed
for that occurrence's predicate frame.

Neighbor Bitmap — ``(atom_id, anchor_id, role_type) -> {neighbor_atom_id}``
with ``role_type`` in ``{'S', 'O', 'N'}``: grouped by the requirement 08
semantic frame membership, each member resolves its anchor through
``(atom_id, anchor_route_frame_pos)`` so a borrowed shared pivot reuses its real
physical anchor and is the one node allowed to appear in two frames. E sources
write ``'N'`` (all other core atoms of their frame), P sources write ``'S'``
(agents) and ``'O'`` (patients). Modifiers never enter Neighbor pairing,
non-shared bits never cross predicate frames, self-loops are never written, and
no empty S/O row is created.

The planner is pure CPU and consumes only the requirement 05 ``AnchorPlan``:
no database, no network, no global state, no re-reading of the tree, no
occurrence duplication, and no event-size special-casing of any kind. The writer
runs on the caller's existing transaction connection only — no pool acquire, no
nested transaction, no commit, no swallowed exception. Both upserts are single
``INSERT ... ON CONFLICT DO UPDATE`` statements whose merge is the
database-side ``rb64_or``, so a Python read-modify-write can never lose a
concurrent update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

# Requirement 06 §7.1/§7.2 SQL. ``{s}`` is the schema placeholder filled by
# the local ``_sql`` helper; the statement shapes are frozen by the docstrings
# below and by the fake-store dispatch in ``tests/noesis_fakes.py``.
_COOCCURRENCE_UPSERT_SQL = (
    "INSERT INTO {s}.cooccurrence_bitmaps (atom_id, anchor_id, event_bitmap) "
    "VALUES ($1, $2, rb64_build(ARRAY[$3]::bigint[])) "
    "ON CONFLICT (atom_id, anchor_id) DO UPDATE "
    "SET event_bitmap = rb64_or({s}.cooccurrence_bitmaps.event_bitmap, EXCLUDED.event_bitmap)"
)
_NEIGHBOR_UPSERT_SQL = (
    "INSERT INTO {s}.neighbor_bitmaps (atom_id, anchor_id, role_type, neighbor_bitmap) "
    "VALUES ($1, $2, $3::text::\"char\", rb64_build($4::bigint[])) "
    "ON CONFLICT (atom_id, anchor_id, role_type) DO UPDATE "
    "SET neighbor_bitmap = rb64_or({s}.neighbor_bitmaps.neighbor_bitmap, EXCLUDED.neighbor_bitmap)"
)

_ALLOWED_ROLES = frozenset({"agent", "patient", "predicate", "modifier"})
_CORE_ROLES = frozenset({"agent", "patient", "predicate"})


class BitmapPlanError(Exception):
    """A fact component cannot be turned into a deterministic bitmap plan.

    Raised for a missing ``atom_id`` / ``frame_pos`` / ``anchor_id``, a
    non-positive Anchor, an illegal role, or a duplicate atom ``pos`` — the
    whole component fails instead of storing half a plan.
    """


class BitmapWriteError(Exception):
    """A bitmap upsert failed inside the fact transaction (whole component rolls back)."""


@dataclass(frozen=True)
class CooccurrenceBitmapWrite:
    """One Cooccurrence row: one bounded event bit per unique (atom, anchor)."""

    atom_id: int
    anchor_id: int
    event_id: int


@dataclass(frozen=True)
class NeighborBitmapWrite:
    """One aggregated Neighbor row: role ``N`` (E), ``S``/``O`` (P) and its bits."""

    atom_id: int
    anchor_id: int
    role_type: Literal["S", "O", "N"]
    neighbor_atom_ids: tuple[int, ...]


@dataclass(frozen=True)
class BitmapWritePlan:
    """Immutable, deduplicated, stably sorted write plan for one fact component."""

    cooccurrences: tuple[CooccurrenceBitmapWrite, ...]
    neighbors: tuple[NeighborBitmapWrite, ...]


def _sql(schema: str, statement: str) -> str:
    return statement.format(s=schema)


def _index_atoms(atoms: Sequence[Any]) -> dict[int, Any]:
    """Index atoms by ``pos``; a duplicate pos means the closure contract broke."""
    by_pos: dict[int, Any] = {}
    for atom in atoms:
        if atom.pos in by_pos:
            raise BitmapPlanError(f"duplicate atom pos {atom.pos}")
        by_pos[atom.pos] = atom
    return by_pos


def plan_bitmap_writes(
    *,
    event_id: int,
    atoms: Sequence[Any],
    anchor_plan: Any,
    atom_ids: Mapping[tuple[str, str], int],
    anchor_routes: Mapping[tuple[int, int], int],
) -> BitmapWritePlan:
    """Plan the two bitmap upserts for one fact component (pure CPU).

    Cooccurrence (requirement 08 §8.1) iterates the physical
    ``anchor_plan.occurrences`` only: every E/P occurrence (modifiers included)
    resolves to its real ``(atom_id, anchor_id)`` bucket, so a borrowed shared
    pivot never produces a second key. Neighbor (requirement 08 §8.2) groups
    ``anchor_plan.semantic_members`` by ``frame_pos``; each member resolves its
    anchor through ``(atom_id, anchor_route_frame_pos)`` — a borrowed member
    reuses the real anchor of its parent-frame occurrence. Within one frame only
    the core roles pair: each E occurrence writes its frame's other core atom IDs
    under ``N`` (self filtered); each P predicate writes all agent IDs under
    ``S`` and all patient IDs under ``O`` (empty side omitted). Neighbor atom IDs
    are deduplicated and ascending; rows are sorted by
    ``(atom_id, anchor_id, role_type)`` for a deterministic lock order.

    Every physical occurrence must be planned exactly once and every
    non-borrowed member must match its atom's role and its own frame; a member
    set that misses an occurrence, a duplicate member, an illegal role, a
    borrowed member that is not a shared agent, or a missing ``atom_id`` /
    ``anchor_id`` raises :class:`BitmapPlanError` immediately instead of storing
    half a plan.
    """
    by_pos = _index_atoms(atoms)

    resolved: list[tuple[int, int, int, str]] = []  # (frame_pos, atom_id, anchor_id, role)
    resolved_positions: set[int] = set()
    for occurrence in anchor_plan.occurrences:
        atom = by_pos.get(occurrence.pos)
        if atom is None:
            raise BitmapPlanError(f"occurrence pos {occurrence.pos} has no atom")
        if occurrence.pos in resolved_positions:
            raise BitmapPlanError(f"duplicate planned occurrence pos {occurrence.pos}")
        resolved_positions.add(occurrence.pos)
        literal = (atom.text, atom.type)
        if occurrence.literal != literal:
            raise BitmapPlanError(
                f"occurrence pos {occurrence.pos} literal {occurrence.literal!r} does not match atom {literal!r}"
            )
        if atom.role not in _ALLOWED_ROLES:
            raise BitmapPlanError(f"atom pos {occurrence.pos} has illegal role {atom.role!r}")
        atom_id = atom_ids.get(occurrence.literal)
        if atom_id is None:
            raise BitmapPlanError(f"missing atom_id for literal {occurrence.literal!r}")
        anchor_id = anchor_routes.get((atom_id, occurrence.frame_pos))
        if not isinstance(anchor_id, int) or anchor_id <= 0:
            raise BitmapPlanError(
                f"missing or non-positive anchor_id for (atom_id={atom_id}, frame_pos={occurrence.frame_pos})"
            )
        resolved.append((occurrence.frame_pos, atom_id, anchor_id, atom.role))

    missing_positions = sorted(set(by_pos) - resolved_positions)
    if missing_positions:
        raise BitmapPlanError(f"anchor plan is missing atom occurrence pos {missing_positions}")

    cooccurrence_keys = sorted({(atom_id, anchor_id) for _, atom_id, anchor_id, _ in resolved})
    cooccurrences = tuple(
        CooccurrenceBitmapWrite(atom_id, anchor_id, event_id) for atom_id, anchor_id in cooccurrence_keys
    )

    physical_members: dict[int, Any] = {}
    frame_members: dict[int, list[tuple[int, int, str]]] = {}
    seen_members: set[tuple[int, int]] = set()
    for member in anchor_plan.semantic_members:
        key = (member.occurrence_pos, member.frame_pos)
        if key in seen_members:
            raise BitmapPlanError(f"duplicate semantic member (occurrence={key[0]}, frame={key[1]})")
        seen_members.add(key)
        atom = by_pos.get(member.occurrence_pos)
        if atom is None:
            raise BitmapPlanError(f"semantic member pos {member.occurrence_pos} has no atom")
        if member.occurrence_pos not in resolved_positions:
            raise BitmapPlanError(f"semantic member pos {member.occurrence_pos} is not a planned occurrence")
        if member.semantic_role not in _ALLOWED_ROLES:
            raise BitmapPlanError(
                f"semantic member pos {member.occurrence_pos} has illegal role {member.semantic_role!r}"
            )
        if member.borrowed:
            if member.semantic_role != "agent" or atom.role not in ("agent", "patient"):
                raise BitmapPlanError(f"borrowed semantic member pos {member.occurrence_pos} is not a shared agent")
        else:
            if member.semantic_role != atom.role or member.frame_pos != member.anchor_route_frame_pos:
                raise BitmapPlanError(
                    f"physical semantic member pos {member.occurrence_pos} does not match its atom role/frame"
                )
            if member.occurrence_pos in physical_members:
                raise BitmapPlanError(f"duplicate physical semantic member pos {member.occurrence_pos}")
            physical_members[member.occurrence_pos] = member
        literal = (atom.text, atom.type)
        atom_id = atom_ids.get(literal)
        if atom_id is None:
            raise BitmapPlanError(f"missing atom_id for literal {literal!r}")
        anchor_id = anchor_routes.get((atom_id, member.anchor_route_frame_pos))
        if not isinstance(anchor_id, int) or anchor_id <= 0:
            raise BitmapPlanError(
                f"missing or non-positive anchor_id for (atom_id={atom_id}, frame_pos={member.anchor_route_frame_pos})"
            )
        frame_members.setdefault(member.frame_pos, []).append((atom_id, anchor_id, member.semantic_role))

    if set(physical_members) != resolved_positions:
        raise BitmapPlanError("semantic members do not cover every atom occurrence")
    for member in anchor_plan.semantic_members:
        if not member.borrowed:
            continue
        physical = physical_members.get(member.occurrence_pos)
        if physical is None or physical.anchor_route_frame_pos != member.anchor_route_frame_pos:
            raise BitmapPlanError("borrowed semantic member must reuse its physical occurrence's anchor route")

    neighbor_sets: dict[tuple[int, int, str], set[int]] = {}
    for frame_pos in sorted(frame_members):
        core = [entry for entry in sorted(frame_members[frame_pos]) if entry[2] in _CORE_ROLES]
        agents = {atom_id for atom_id, _, role in core if role == "agent"}
        patients = {atom_id for atom_id, _, role in core if role == "patient"}
        for atom_id, anchor_id, role in core:
            if role == "predicate":
                source_agents = agents - {atom_id}
                source_patients = patients - {atom_id}
                if source_agents:
                    neighbor_sets.setdefault((atom_id, anchor_id, "S"), set()).update(source_agents)
                if source_patients:
                    neighbor_sets.setdefault((atom_id, anchor_id, "O"), set()).update(source_patients)
            else:
                others = {other_id for other_id, _, _ in core if other_id != atom_id}
                if others:
                    neighbor_sets.setdefault((atom_id, anchor_id, "N"), set()).update(others)

    neighbors = tuple(
        NeighborBitmapWrite(atom_id, anchor_id, role_type, tuple(sorted(neighbor_ids)))
        for (atom_id, anchor_id, role_type), neighbor_ids in sorted(neighbor_sets.items())
    )
    return BitmapWritePlan(cooccurrences=cooccurrences, neighbors=neighbors)


async def apply_bitmap_writes(conn: Any, schema: str, plan: BitmapWritePlan) -> None:
    """Execute the plan on the caller's open fact transaction connection.

    One statement per target row, in plan order: cooccurrence first (sorted by
    ``(atom_id, anchor_id)``), then neighbor (sorted by
    ``(atom_id, anchor_id, role_type)``) — a stable lock order for concurrent
    events hitting overlapping buckets. The merge is the database-side
    ``rb64_or`` inside ``ON CONFLICT DO UPDATE``; no SELECT-then-UPDATE path
    exists. The connection is never acquired, committed, or nested; database
    errors are re-raised as :class:`BitmapWriteError` with the original cause
    preserved, and cancellation propagates untouched (``CancelledError`` is a
    ``BaseException``).
    """
    try:
        for write in plan.cooccurrences:
            await conn.execute(
                _sql(schema, _COOCCURRENCE_UPSERT_SQL),
                write.atom_id,
                write.anchor_id,
                write.event_id,
            )
        for write in plan.neighbors:
            await conn.execute(
                _sql(schema, _NEIGHBOR_UPSERT_SQL),
                write.atom_id,
                write.anchor_id,
                write.role_type,
                list(write.neighbor_atom_ids),
            )
    except Exception as error:
        raise BitmapWriteError(f"bitmap write failed: {type(error).__name__}") from error
