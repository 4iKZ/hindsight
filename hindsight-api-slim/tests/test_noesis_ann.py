"""Noesis ANN Top-100 recall contract tests (requirement 04 Tasks 1–3).

Offline only: a scriptable fake connection. No real PostgreSQL and no bge HTTP.
Tasks 1–2 freeze types and validation doors. Task 3 freezes E/P Top-100 SQL
and row conversion. Task 4 freezes SET LOCAL knobs in a short read-only
transaction and proves recall still works with no IVFFlat index.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import FrozenInstanceError, fields

import pytest

from hindsight_api.engine.retain import noesis_ann as ann_module
from hindsight_api.engine.retain.noesis_ann import (
    AnnCandidate,
    AnnProfileUnavailable,
    AnnRecallError,
    AnnSourceIneligible,
    AnnSourceNotFound,
    recall_ann_candidates,
)

DIM = 1024


def _vector(seed: float = 0.01, dim: int = DIM) -> list[float]:
    return [seed] * dim


def _ready_profile(**overrides):
    profile = {
        "model_name": "bge-m3",
        "model_revision": "bge-m3-1024-v1",
        "dimension": DIM,
        "status": "ready",
    }
    profile.update(overrides)
    return profile


def _atom(
    atom_id: int,
    *,
    text: str = "苹果手机",
    atom_type: str = "E",
    status: str = "active",
    embedding,
) -> dict:
    return {
        "atom_id": atom_id,
        "text": text,
        "atom_type": atom_type,
        "status": status,
        "embedding": embedding,
    }


def _cosine_distance(left, right) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 1.0
    return 1.0 - (dot / (norm_left * norm_right))


class _AnnTx:
    def __init__(self, conn: "_AnnConn") -> None:
        self._conn = conn

    async def __aenter__(self):
        self._conn._tx_depth += 1
        self._conn.calls.append(("begin", "", ()))
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self._conn._local_settings.clear()
        self._conn._tx_depth -= 1
        self._conn.calls.append(("commit" if exc_type is None else "rollback", "", ()))
        return False


class _AnnConn:
    """Minimal asyncpg-shaped fake: profile, source, and candidate reads."""

    def __init__(self, *, profile=None, atoms=None, distance_overrides=None, indexes=None) -> None:
        self.profile = profile
        self.atoms = dict(atoms or {})
        self.distance_overrides = dict(distance_overrides or {})
        self.indexes = list(indexes or [])
        self.calls: list[tuple[str, str, tuple]] = []
        self.transaction_kwargs: list[dict] = []
        self._tx_depth = 0
        self._local_settings: dict[str, str] = {}
        self._session_settings: dict[str, str] = {}

    def transaction(self, **kwargs):
        self.transaction_kwargs.append(dict(kwargs))
        return _AnnTx(self)

    async def fetchrow(self, sql: str, *args):
        self.calls.append(("fetchrow", sql, args))
        if ".embedding_profiles" in sql:
            return None if self.profile is None else dict(self.profile)
        if _is_candidate_sql(sql):
            raise AssertionError("candidate SQL must use fetch, not fetchrow")
        if ".atoms" in sql:
            atom_id = args[0]
            row = self.atoms.get(atom_id)
            return None if row is None else dict(row)
        raise AssertionError(f"unexpected fetchrow SQL: {sql}")

    async def fetch(self, sql: str, *args):
        self.calls.append(("fetch", sql, args))
        if _is_candidate_sql(sql):
            return self._candidates(sql, args)
        if ".embedding_profiles" in sql:
            return [] if self.profile is None else [dict(self.profile)]
        raise AssertionError(f"unexpected fetch SQL: {sql}")

    async def execute(self, sql: str, *args):
        self.calls.append(("execute", sql, args))
        compact = " ".join(sql.split())
        upper = compact.upper()
        if upper.startswith("SET LOCAL "):
            if self._tx_depth == 0:
                raise AssertionError("SET LOCAL must run inside a transaction")
            name, _, value = compact[10:].partition("=")
            self._local_settings[name.strip().lower()] = value.strip().strip("'")
            return "SET"
        if upper.startswith("SET "):
            name, _, value = compact[4:].partition("=")
            self._session_settings[name.strip().lower()] = value.strip().strip("'")
            return "SET"
        return "OK"

    def _candidates(self, sql: str, args: tuple) -> list[dict]:
        source_atom_id, limit = args[0], args[1]
        source = self.atoms.get(source_atom_id)
        assert source is not None and source["embedding"] is not None
        required_type = "E" if "atom_type = 'E'" in sql else None
        if required_type is None and "atom_type = 'P'" in sql:
            required_type = "P"
        if required_type is None:
            raise AssertionError("candidate SQL must use a frozen E or P literal")
        rows = []
        for atom_id, atom in self.atoms.items():
            if atom_id == source_atom_id:
                continue
            if atom["status"] != "active" or atom["atom_type"] != required_type:
                continue
            if atom["embedding"] is None:
                continue
            distance = self.distance_overrides.get(atom_id)
            if distance is None:
                distance = _cosine_distance(source["embedding"], atom["embedding"])
            rows.append(
                {
                    "atom_id": atom_id,
                    "text": atom["text"],
                    "atom_type": atom["atom_type"],
                    "distance": distance,
                    "similarity": 1.0 - distance if isinstance(distance, (int, float)) else distance,
                }
            )
        rows.sort(key=lambda row: ((row["distance"] if math.isfinite(row["distance"]) else 0.0), row["atom_id"]))
        return rows[: int(limit)]


def _is_candidate_sql(sql: str) -> bool:
    return "<=>" in sql or "WITH nearest" in sql


def _mixed_atoms() -> dict[int, dict]:
    return {
        1: _atom(1, text="苹果手机", embedding=[1.0, 0.0]),
        2: _atom(2, text="iPhone", embedding=[0.95, 0.05]),
        3: _atom(3, text="智能手机", embedding=[0.7, 0.3]),
        4: _atom(4, text="旧手机", status="inactive", embedding=[0.99, 0.01]),
        5: _atom(5, text="无向量", embedding=None),
        6: _atom(6, text="周末计划", atom_type="G", embedding=None),
        10: _atom(10, text="买", atom_type="P", embedding=[1.0, 0.0]),
        11: _atom(11, text="购买", atom_type="P", embedding=[0.95, 0.05]),
        12: _atom(12, text="出售", atom_type="P", embedding=[0.2, 0.8]),
    }


async def _recall(conn, **kwargs):
    return await recall_ann_candidates(conn, schema="noesis_core", **kwargs)


# ---------------------------------------------------------------------------
# AnnCandidate + exception types
# ---------------------------------------------------------------------------


def test_ann_candidate_fields_and_frozen():
    candidate = AnnCandidate(
        atom_id=7,
        text="苹果手机",
        atom_type="E",
        distance=0.1,
        similarity=0.9,
    )
    assert [field.name for field in fields(AnnCandidate)] == [
        "atom_id",
        "text",
        "atom_type",
        "distance",
        "similarity",
    ]
    assert candidate.atom_id == 7
    assert candidate.text == "苹果手机"
    assert candidate.atom_type == "E"
    assert candidate.distance == 0.1
    assert candidate.similarity == 0.9
    assert not hasattr(candidate, "support_count")
    with pytest.raises(FrozenInstanceError):
        candidate.atom_id = 8  # type: ignore[misc]


def test_ann_exception_hierarchy():
    assert issubclass(AnnSourceNotFound, AnnRecallError)
    assert issubclass(AnnSourceIneligible, AnnRecallError)
    assert issubclass(AnnProfileUnavailable, AnnRecallError)


# ---------------------------------------------------------------------------
# Source existence / eligibility
# ---------------------------------------------------------------------------


async def test_missing_source_raises_not_found():
    conn = _AnnConn(profile=_ready_profile(), atoms={})
    with pytest.raises(AnnSourceNotFound):
        await _recall(conn, source_atom_id=999)


@pytest.mark.parametrize(
    "atom",
    [
        _atom(1, status="inactive", embedding=_vector()),
        _atom(2, text="周末计划", atom_type="G", embedding=None),
        _atom(3, embedding=None),
    ],
    ids=["inactive", "type_g", "null_embedding"],
)
async def test_ineligible_source_raises(atom):
    conn = _AnnConn(profile=_ready_profile(), atoms={atom["atom_id"]: atom})
    with pytest.raises(AnnSourceIneligible):
        await _recall(conn, source_atom_id=atom["atom_id"])


# ---------------------------------------------------------------------------
# Identity profile gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile",
    [
        None,
        _ready_profile(status="rebuilding"),
        _ready_profile(dimension=768),
    ],
    ids=["missing", "rebuilding", "dimension_mismatch"],
)
async def test_profile_unavailable_raises(profile):
    source = _atom(1, embedding=_vector())
    conn = _AnnConn(profile=profile, atoms={1: source})
    with pytest.raises(AnnProfileUnavailable):
        await _recall(conn, source_atom_id=1)


# ---------------------------------------------------------------------------
# Input validation: must not become an empty candidate list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema",
    ["bad name", "noesis-core", "", "1core", 'core"; DROP TABLE atoms', "noesis.core", "表"],
)
async def test_illegal_schema_raises(schema):
    conn = _AnnConn(profile=_ready_profile(), atoms={1: _atom(1, embedding=_vector())})
    with pytest.raises(ValueError):
        await recall_ann_candidates(conn, schema=schema, source_atom_id=1)


@pytest.mark.parametrize("source_atom_id", [0, -1, True, False, 1.5, "1", None])
async def test_illegal_source_atom_id_raises(source_atom_id):
    conn = _AnnConn(profile=_ready_profile(), atoms={1: _atom(1, embedding=_vector())})
    with pytest.raises((TypeError, ValueError)):
        await _recall(conn, source_atom_id=source_atom_id)


@pytest.mark.parametrize("limit", [0, 101, -1, 1000])
async def test_illegal_limit_raises(limit):
    conn = _AnnConn(profile=_ready_profile(), atoms={1: _atom(1, embedding=_vector())})
    with pytest.raises(ValueError):
        await _recall(conn, source_atom_id=1, limit=limit)


# ---------------------------------------------------------------------------
# Task 3: E/P Top-100 frozen query
# ---------------------------------------------------------------------------


def test_frozen_sql_templates_are_literal_e_and_p():
    e_sql = ann_module._ANN_QUERY_BY_TYPE["E"]
    p_sql = ann_module._ANN_QUERY_BY_TYPE["P"]
    assert "atom_type = 'E'" in e_sql
    assert "atom_type = 'P'" not in e_sql
    assert "atom_type = 'P'" in p_sql
    assert "atom_type = 'E'" not in p_sql
    assert "support_count" not in e_sql and "support_count" not in p_sql
    assert "ORDER BY a.embedding <=> s.embedding" in e_sql
    assert "ORDER BY distance + 0, atom_id" in e_sql
    assert "MATERIALIZED" in e_sql
    assert "{atom_type}" not in e_sql
    assert p_sql == e_sql.replace("atom_type = 'E'", "atom_type = 'P'")


async def test_e_source_returns_only_e():
    conn = _AnnConn(profile=_ready_profile(), atoms=_mixed_atoms())
    candidates = await _recall(conn, source_atom_id=1)
    assert candidates
    assert all(item.atom_type == "E" for item in candidates)
    assert {item.atom_id for item in candidates} == {2, 3}
    assert [item.atom_id for item in candidates] == [2, 3]


async def test_p_source_returns_only_p():
    conn = _AnnConn(profile=_ready_profile(), atoms=_mixed_atoms())
    candidates = await _recall(conn, source_atom_id=10)
    assert candidates
    assert all(item.atom_type == "P" for item in candidates)
    assert {item.atom_id for item in candidates} == {11, 12}


async def test_candidates_exclude_self_inactive_g_and_null():
    conn = _AnnConn(profile=_ready_profile(), atoms=_mixed_atoms())
    candidates = await _recall(conn, source_atom_id=1)
    ids = {item.atom_id for item in candidates}
    assert 1 not in ids
    assert 4 not in ids
    assert 5 not in ids
    assert 6 not in ids
    assert 10 not in ids


async def test_fewer_than_limit_returns_all_legal_candidates():
    conn = _AnnConn(profile=_ready_profile(), atoms=_mixed_atoms())
    candidates = await _recall(conn, source_atom_id=1, limit=100)
    assert len(candidates) == 2


def _many_e_atoms(count: int) -> dict[int, dict]:
    atoms = {1: _atom(1, text="source", embedding=[1.0, 0.0])}
    for atom_id in range(2, count + 2):
        atoms[atom_id] = _atom(atom_id, text=f"e{atom_id}", embedding=[1.0, 0.0])
    return atoms


async def test_more_than_limit_is_truncated_to_100():
    conn = _AnnConn(profile=_ready_profile(), atoms=_many_e_atoms(101))
    candidates = await _recall(conn, source_atom_id=1)
    assert len(candidates) == 100
    assert [item.atom_id for item in candidates] == list(range(2, 102))


async def test_legal_limit_one_and_one_hundred():
    conn = _AnnConn(profile=_ready_profile(), atoms=_many_e_atoms(101))
    one = await _recall(conn, source_atom_id=1, limit=1)
    hundred = await _recall(conn, source_atom_id=1, limit=100)
    assert len(one) == 1
    assert one[0].atom_id == 2
    assert len(hundred) == 100


async def test_similarity_is_one_minus_distance():
    conn = _AnnConn(profile=_ready_profile(), atoms=_mixed_atoms())
    candidates = await _recall(conn, source_atom_id=1)
    assert candidates
    for item in candidates:
        assert item.similarity == pytest.approx(1.0 - item.distance)
        assert math.isfinite(item.distance)
        assert math.isfinite(item.similarity)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
async def test_non_finite_distance_is_rejected(bad):
    atoms = {
        1: _atom(1, embedding=[1.0, 0.0]),
        2: _atom(2, text="坏", embedding=[0.9, 0.1]),
    }
    conn = _AnnConn(profile=_ready_profile(), atoms=atoms, distance_overrides={2: bad})
    with pytest.raises(AnnRecallError):
        await _recall(conn, source_atom_id=1)


async def test_equal_distance_sorts_by_atom_id():
    atoms = {
        1: _atom(1, embedding=[1.0, 0.0]),
        9: _atom(9, text="晚", embedding=[1.0, 0.0]),
        3: _atom(3, text="早", embedding=[1.0, 0.0]),
    }
    conn = _AnnConn(profile=_ready_profile(), atoms=atoms)
    candidates = await _recall(conn, source_atom_id=1)
    assert [item.atom_id for item in candidates] == [3, 9]
    assert candidates[0].distance == pytest.approx(candidates[1].distance)


async def test_recall_is_read_only_and_does_not_call_bge():
    conn = _AnnConn(profile=_ready_profile(), atoms=_mixed_atoms())
    await _recall(conn, source_atom_id=1)
    forbidden = ("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "REINDEX")
    for _method, sql, _args in conn.calls:
        leading = sql.lstrip().upper()
        assert not leading.startswith(forbidden)
    assert not hasattr(ann_module, "NoesisIdentityClient")
    text = inspect.getsource(ann_module)
    assert "NoesisIdentityClient" not in text
    assert "/normalize/" not in text


# ---------------------------------------------------------------------------
# Task 4: SET LOCAL + same SQL with no index
# ---------------------------------------------------------------------------


def _set_local_calls(conn: _AnnConn) -> list[str]:
    statements = []
    for method, sql, _args in conn.calls:
        if method != "execute":
            continue
        compact = " ".join(sql.split())
        if compact.upper().startswith("SET LOCAL "):
            statements.append(compact)
    return statements


async def test_set_local_uses_frozen_ivfflat_knobs():
    conn = _AnnConn(profile=_ready_profile(), atoms=_mixed_atoms())
    await _recall(conn, source_atom_id=1)
    assert _set_local_calls(conn) == [
        "SET LOCAL ivfflat.probes = 10",
        "SET LOCAL ivfflat.iterative_scan = relaxed_order",
        "SET LOCAL ivfflat.max_probes = 100",
    ]
    assert not any(
        method == "execute" and " ".join(sql.split()).upper().startswith("SET ")
        and not " ".join(sql.split()).upper().startswith("SET LOCAL ")
        for method, sql, _args in conn.calls
    )


async def test_set_local_and_candidate_select_share_readonly_transaction():
    conn = _AnnConn(profile=_ready_profile(), atoms=_mixed_atoms())
    await _recall(conn, source_atom_id=1)
    assert conn.transaction_kwargs
    assert any(kwargs.get("readonly") is True for kwargs in conn.transaction_kwargs)
    methods = [method for method, _sql, _args in conn.calls]
    begin = methods.index("begin")
    commit = methods.index("commit")
    window = conn.calls[begin + 1 : commit]
    set_locals = [
        " ".join(sql.split())
        for method, sql, _args in window
        if method == "execute" and " ".join(sql.split()).upper().startswith("SET LOCAL ")
    ]
    fetches = [sql for method, sql, _args in window if method == "fetch" and _is_candidate_sql(sql)]
    assert set_locals == [
        "SET LOCAL ivfflat.probes = 10",
        "SET LOCAL ivfflat.iterative_scan = relaxed_order",
        "SET LOCAL ivfflat.max_probes = 100",
    ]
    assert fetches
    assert methods.index("fetch") > begin
    assert methods.index("fetch") < commit


async def test_ivfflat_settings_do_not_leak_after_transaction():
    conn = _AnnConn(profile=_ready_profile(), atoms=_mixed_atoms())
    await _recall(conn, source_atom_id=1)
    assert conn._tx_depth == 0
    assert conn._local_settings == {}
    assert not any(name.startswith("ivfflat.") for name in conn._session_settings)


async def test_recall_works_without_ivfflat_index():
    conn = _AnnConn(profile=_ready_profile(), atoms=_mixed_atoms(), indexes=[])
    candidates = await _recall(conn, source_atom_id=1)
    assert [item.atom_id for item in candidates] == [2, 3]
    assert not any("idx_atoms_embedding_ivfflat" in sql for _method, sql, _args in conn.calls)
