"""Noesis ANN Top-100 recall (requirement 04).

Tasks 1–4: public types, validation doors, frozen E/P candidate SQL, and
SET LOCAL knobs in a short read-only transaction. IVFFlat index lifecycle
belongs to later tasks. This module only reads; it never writes
atoms/events and never calls bge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from .noesis_ingest import validate_schema_identifier

_READY_STATUS = "ready"
_IDENTITY_DIMENSION = 1024
_MAX_LIMIT = 100

_PROFILE_SELECT = (
    "SELECT model_name, model_revision, dimension, status "
    "FROM {s}.embedding_profiles "
    "WHERE embedding_kind = 'identity'"
)
_SOURCE_SELECT = (
    "SELECT atom_id, text, atom_type, status, embedding "
    "FROM {s}.atoms "
    "WHERE atom_id = $1"
)
_ANN_QUERY_E = """
WITH nearest AS MATERIALIZED (
    SELECT
        a.atom_id,
        a.text,
        a.atom_type,
        a.embedding <=> s.embedding AS distance
    FROM {s}.atoms AS a
    CROSS JOIN (
        SELECT embedding
        FROM {s}.atoms
        WHERE atom_id = $1
    ) AS s
    WHERE a.atom_id <> $1
      AND a.status = 'A'
      AND a.atom_type = 'E'
      AND a.embedding IS NOT NULL
    ORDER BY a.embedding <=> s.embedding
    LIMIT $2
)
SELECT
    atom_id,
    text,
    atom_type,
    distance,
    1.0 - distance AS similarity
FROM nearest
ORDER BY distance + 0, atom_id
"""
_ANN_QUERY_BY_TYPE = {
    "E": _ANN_QUERY_E,
    "P": _ANN_QUERY_E.replace("atom_type = 'E'", "atom_type = 'P'"),
}
_IVFFLAT_LOCAL = (
    "SET LOCAL ivfflat.probes = 10",
    "SET LOCAL ivfflat.iterative_scan = relaxed_order",
    "SET LOCAL ivfflat.max_probes = 100",
)


@dataclass(frozen=True)
class AnnCandidate:
    atom_id: int
    text: str
    atom_type: Literal["E", "P"]
    distance: float
    similarity: float


class AnnRecallError(Exception):
    """Base class for deterministic ANN recall failures."""


class AnnSourceNotFound(AnnRecallError):
    """The requested atom_id does not exist."""


class AnnSourceIneligible(AnnRecallError):
    """The source is deprecated (D), G, or has no Identity Vector."""


class AnnProfileUnavailable(AnnRecallError):
    """The identity embedding profile is missing or not ready."""


def _sql(schema: str, statement: str) -> str:
    return statement.format(s=schema)


def _require_schema(schema: str) -> str:
    if not validate_schema_identifier(schema):
        raise ValueError(f"invalid noesis schema identifier: {schema!r}")
    return schema


def _require_source_atom_id(source_atom_id: Any) -> int:
    if type(source_atom_id) is not int or source_atom_id <= 0:
        raise ValueError(f"source_atom_id must be a positive int, got {source_atom_id!r}")
    return source_atom_id


def _require_limit(limit: Any) -> int:
    if type(limit) is not int or limit < 1 or limit > _MAX_LIMIT:
        raise ValueError(f"limit must be an int in 1–{_MAX_LIMIT}, got {limit!r}")
    return limit


def _row_value(row: Any, key: str) -> Any:
    return row[key]


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AnnRecallError(f"candidate {name} is not a number") from error
    if not math.isfinite(number):
        raise AnnRecallError(f"candidate {name} is not finite")
    return number


def _candidate_from_row(row: Any) -> AnnCandidate:
    atom_type = str(_row_value(row, "atom_type") or "").strip()
    if atom_type not in ("E", "P"):
        raise AnnRecallError(f"candidate atom_type must be E or P, got {atom_type!r}")
    distance = _finite_float(_row_value(row, "distance"), "distance")
    similarity = 1.0 - distance
    if not math.isfinite(similarity):
        raise AnnRecallError("candidate similarity is not finite")
    return AnnCandidate(
        atom_id=int(_row_value(row, "atom_id")),
        text=str(_row_value(row, "text")),
        atom_type=atom_type,  # type: ignore[arg-type]
        distance=distance,
        similarity=similarity,
    )


async def recall_ann_candidates(
    conn,
    *,
    schema: str,
    source_atom_id: int,
    limit: int = 100,
) -> list[AnnCandidate]:
    schema = _require_schema(schema)
    source_atom_id = _require_source_atom_id(source_atom_id)
    limit = _require_limit(limit)

    profile = await conn.fetchrow(_sql(schema, _PROFILE_SELECT))
    if profile is None:
        raise AnnProfileUnavailable("identity embedding profile is missing")
    status = _row_value(profile, "status")
    try:
        dimension = int(_row_value(profile, "dimension"))
    except (TypeError, ValueError) as error:
        raise AnnProfileUnavailable("identity embedding profile dimension is invalid") from error
    if status != _READY_STATUS or dimension != _IDENTITY_DIMENSION:
        raise AnnProfileUnavailable(
            f"identity embedding profile is not ready (status={status!r}, dimension={dimension})"
        )

    source = await conn.fetchrow(_sql(schema, _SOURCE_SELECT), source_atom_id)
    if source is None:
        raise AnnSourceNotFound(f"atom_id {source_atom_id} does not exist")
    atom_type = str(_row_value(source, "atom_type") or "").strip()
    source_status = _row_value(source, "status")
    embedding = _row_value(source, "embedding")
    if source_status != "A" or atom_type not in ("E", "P") or embedding is None:
        raise AnnSourceIneligible(
            f"atom_id {source_atom_id} is not an active (status='A') E/P atom with an Identity Vector"
        )

    query = _ANN_QUERY_BY_TYPE[atom_type]
    async with conn.transaction(readonly=True):
        for statement in _IVFFLAT_LOCAL:
            await conn.execute(statement)
        rows = await conn.fetch(_sql(schema, query), source_atom_id, limit)
    return [_candidate_from_row(row) for row in rows]
