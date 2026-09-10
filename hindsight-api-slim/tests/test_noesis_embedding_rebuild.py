"""Noesis shared embedding-space rebuild tests (requirement 05A §6/§11.3/§11.4).

The rebuild state machine runs against the injected FakeStore + FakeEmbeddingClient;
no real database, no real network. Each test drives ``run_embedding_rebuild``
directly and asserts the documented invariants: advisory lock, rebuilding
status, no live mutation on failure, historical reconstruction from
``events.data.component``, deterministic EMA replay, orphan fail-closed, ANN
index collaboration, and one atomic atoms + anchors + ANN + profile cutover.
"""

from __future__ import annotations

import json

import pytest

from hindsight_api.engine.retain.noesis_embedding_rebuild import (
    BuildSpec,
    _replay_anchor_centroids,
    run_embedding_rebuild,
)
from tests.noesis_fakes import (
    FakeEmbeddingClient,
    FakePool,
    FakeStore,
    fake_context_vector,
    fake_identity_vector,
)

FROZEN_ANN = (
    "CREATE INDEX idx_atoms_embedding_ivfflat ON noesis_core.atoms "
    "USING ivfflat (embedding vector_cosine_ops)"
)


def _spec(model: str = "bge-m3", revision: str = "bge-m3-1024-v1", dimension: int = 1024) -> BuildSpec:
    return BuildSpec(model=model, revision=revision, dimension=dimension)


def _component(agent: str, predicate: str, patient: str) -> dict:
    return {
        "utterance_type": "fact",
        "atoms": [
            {"pos": 1, "text": agent, "type": "E", "role": "agent", "target_occ": 2, "resolved": None},
            {"pos": 2, "text": predicate, "type": "P", "role": "predicate", "target_occ": None, "resolved": None},
            {"pos": 3, "text": patient, "type": "E", "role": "patient", "target_occ": 2, "resolved": None},
        ],
        "tree": {
            "predicate": predicate,
            "agent": [{"text": agent, "modifier": [], "implied": False}],
            "patient": [{"text": patient, "modifier": [], "implied": False}],
            "modifier": [],
            "nested": [],
            "conditional": [],
        },
    }


def _ready_profile(revision: str = "bge-m3-1024-v1") -> dict:
    return {
        "embedding_kind": "identity",
        "model_name": "bge-m3",
        "model_revision": revision,
        "dimension": 1024,
        "status": "ready",
    }


async def _run(store, client=None, *, spec=None, repair_null_only=False, force_full=False, schema="noesis_core"):
    return await run_embedding_rebuild(
        pool=FakePool(store),
        schema=schema,
        client=client or FakeEmbeddingClient(),
        spec=spec or _spec(),
        repair_null_only=repair_null_only,
        force_full=force_full,
    )


def _store_with_history() -> FakeStore:
    """Three events sharing the atom 修复/服务器; per-atom anchors preserved.

    atom ids: 张三=1 李四=2 王五=3 修复=4 服务器=5
    anchors:  张三=101 李四=102 王五=103 修复=104 服务器=105
    """
    store = FakeStore()
    store.profile = _ready_profile()
    for text, atom_type, atom_id in (
        ("张三", "E", 1),
        ("李四", "E", 2),
        ("王五", "E", 3),
        ("修复", "P", 4),
        ("服务器", "E", 5),
    ):
        store.atoms[(text, atom_type)] = {"atom_id": atom_id, "embedding": fake_identity_vector(text, atom_type)}
    # Pre-existing anchors with their old-generation centroids and counts.
    store.seed_anchor(1, (0.0,) * 1024, total_count=7)
    store.seed_anchor(2, (0.0,) * 1024, total_count=7)
    store.seed_anchor(3, (0.0,) * 1024, total_count=7)
    store.seed_anchor(4, (0.0,) * 1024, total_count=7)
    store.seed_anchor(5, (0.0,) * 1024, total_count=7)
    assert sorted(store.anchors) == [101, 102, 103, 104, 105]
    for event_id, agent, agent_atom, agent_anchor in (
        (1, "张三", 1, 101),
        (2, "李四", 2, 102),
        (3, "王五", 3, 103),
    ):
        store.events[event_id] = {
            "event_id": event_id,
            "event_time": None,
            "data": {"component": _component(agent, "修复", "服务器")},
            "source": "L",
            "category": "R",
        }
        store.event_atoms.extend(
            [
                (event_id, 1, agent_atom, "A", 2, agent_anchor),
                (event_id, 2, 4, "R", None, 104),
                (event_id, 3, 5, "P", 2, 105),
            ]
        )
    return store


# ---------------------------------------------------------------------------
# §6.3 lock + §6.4 profile state machine
# ---------------------------------------------------------------------------

async def test_second_rebuild_refused_when_lock_held():
    store = _store_with_history()
    store.advisory_lock_available = False
    assert await _run(store) == 2
    assert store.profile["status"] == "ready"


async def test_ready_same_generation_is_a_noop_without_force():
    store = _store_with_history()
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v1")) == 0
    # No staging ran, no anchor touched: the old generation is still live.
    assert store.anchors[105]["total_count"] == 7
    assert not any("CREATE TEMP TABLE" in call[1] for call in store.calls)


async def test_rebuild_runs_even_when_generation_matches_with_force():
    store = _store_with_history()
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v1"), force_full=True) == 0
    assert store.anchors[105]["total_count"] == 3


async def test_jsonb_string_event_data_is_decoded():
    store = _store_with_history()
    # asyncpg may hand back JSONB as a string unless a codec is registered.
    store.events[1]["data"] = json.dumps(store.events[1]["data"], ensure_ascii=False)
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 0
    assert store.anchors[105]["total_count"] == 3


async def test_client_not_ready_aborts_without_mutation():
    store = _store_with_history()
    assert await _run(store, FakeEmbeddingClient(health="config_invalid")) == 4
    assert store.profile["status"] == "ready"
    assert store.anchors[105]["total_count"] == 7


# ---------------------------------------------------------------------------
# §6.8/§6.9/§6.10 rebuild content: atoms + anchors + deterministic EMA
# ---------------------------------------------------------------------------

async def test_rebuild_replaces_atom_vectors_and_anchor_centroids():
    store = _store_with_history()
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 0

    assert store.profile["model_revision"] == "bge-m3-1024-v2"
    assert store.profile["status"] == "ready"
    for (text, atom_type), atom in store.atoms.items():
        assert atom["embedding"] == fake_identity_vector(text, atom_type)

    # Anchor 105 (服务器) has three samples with three distinct frame contexts,
    # replayed in (anchor_id, event_id, predicate_pos) order.
    c1 = fake_context_vector("张三 修复 服务器")
    c2 = fake_context_vector("李四 修复 服务器")
    c3 = fake_context_vector("王五 修复 服务器")
    expected = tuple(0.9 * (0.9 * a + 0.1 * b) + 0.1 * c for a, b, c in zip(c1, c2, c3))
    assert store.anchors[105]["total_count"] == 3
    assert store.anchors[105]["centroid"] == expected
    # Anchor 101 (张三) has a single sample: centroid == that context.
    assert store.anchors[101]["total_count"] == 1
    assert store.anchors[101]["centroid"] == c1


async def test_rebuild_preserves_event_atoms_and_event_data():
    store = _store_with_history()
    event_atoms_before = list(store.event_atoms)
    events_before = {event_id: dict(event["data"]) for event_id, event in store.events.items()}

    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 0

    assert store.event_atoms == event_atoms_before
    for event_id, event in store.events.items():
        assert event["data"] == events_before[event_id]


async def test_rebuild_never_writes_context_embedding_or_routes_anchors():
    store = _store_with_history()
    before = {anchor_id: dict(anchor) for anchor_id, anchor in store.anchors.items()}
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 0

    # Anchor ids are preserved (no new anchors, no re-routing).
    assert sorted(store.anchors) == sorted(before)
    for anchor_id, anchor in store.anchors.items():
        assert anchor["atom_id"] == before[anchor_id]["atom_id"]
        assert anchor["status"] == before[anchor_id]["status"]
    # No event-level context vector was ever written.
    assert not any("context_embedding" in call[1] for call in store.calls)


async def test_g_atoms_are_never_encoded():
    store = _store_with_history()
    store.atoms[("周末", "G")] = {"atom_id": 900, "embedding": None}
    client = FakeEmbeddingClient()
    assert await _run(store, client, spec=_spec(revision="bge-m3-1024-v2")) == 0
    assert ("周末", "G") not in client.identity_calls
    assert store.atoms[("周末", "G")]["embedding"] is None


async def test_keyset_pagination_uses_256_row_pages():
    store = FakeStore()
    store.profile = _ready_profile()
    for index in range(1, 258):  # 257 atoms -> two pages
        store.atoms[(f"词{index}", "E")] = {"atom_id": index, "embedding": None}

    client = FakeEmbeddingClient()
    assert await _run(store, client, spec=_spec(revision="bge-m3-1024-v2")) == 0
    assert len(store.atoms) == 257
    assert len(client.identity_calls) == 257

    pages = [call for call in store.calls if "atom_id > $1" in call[1]]
    # Two data pages (256 + 1) plus the empty terminator page.
    assert [call[2][0] for call in pages] == [0, 256, 257]
    assert "LIMIT 256" in pages[0][1]


async def test_anchor_replay_uses_bounded_keyset_pages_without_resetting_ema():
    store = FakeStore()
    context = fake_context_vector("shared context")
    for event_id in range(1, 258):
        store.embedding_anchor_sample_stage[(101, event_id, 2)] = True
        store.embedding_context_stage[(event_id, 2)] = context

    assert await _replay_anchor_centroids(store, _spec()) == 0

    centroid, total_count = store.embedding_anchor_stage[101]
    assert centroid == pytest.approx(context)
    assert total_count == 257
    pages = [
        call
        for call in store.calls
        if "noesis_embedding_anchor_sample_stage" in call[1] and "JOIN" in call[1]
    ]
    assert [call[2] for call in pages] == [
        (-1, -1, -1),
        (101, 256, 2),
        (101, 257, 2),
    ]
    assert all("LIMIT 256" in call[1] for call in pages)


# ---------------------------------------------------------------------------
# §6.11 orphan anchors + §4.4 history consistency fail-closed
# ---------------------------------------------------------------------------

async def test_orphan_anchor_makes_the_rebuild_fail_closed():
    store = _store_with_history()
    store.seed_anchor(1, (0.0,) * 1024, total_count=1)  # a sixth anchor with no sample

    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 8
    assert store.profile["status"] == "rebuilding"
    assert store.anchors[105]["total_count"] == 7  # live centroids untouched


async def test_history_rejected_when_anchor_belongs_to_another_atom():
    store = _store_with_history()
    # Point 服务器's occurrence at 修复's anchor.
    store.event_atoms = [
        (event_id, occ, atom_id, role, target, 104 if atom_id == 5 else anchor)
        for (event_id, occ, atom_id, role, target, anchor) in store.event_atoms
    ]
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 7
    assert store.profile["status"] == "rebuilding"


async def test_history_rejected_when_occurrence_missing_from_component():
    store = _store_with_history()
    store.event_atoms.append((1, 99, 1, "A", 2, 101))  # no component pos 99
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 7


async def test_history_rejected_on_null_anchor():
    store = _store_with_history()
    store.event_atoms[0] = (1, 1, 1, "A", 2, None)
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 7


async def test_history_rejected_when_component_is_not_a_valid_closure():
    store = _store_with_history()
    store.events[1]["data"] = {"component": {"utterance_type": "fact", "atoms": [], "tree": {}}}
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 7


# ---------------------------------------------------------------------------
# §6.12 ANN index collaboration
# ---------------------------------------------------------------------------

async def test_no_ann_index_is_allowed_and_never_created():
    store = _store_with_history()
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 0
    assert not any(call[1].startswith("REINDEX INDEX") for call in store.calls)


async def test_frozen_ivfflat_index_is_reindexed_in_final_transaction():
    store = _store_with_history()
    store.pg_indexes = [{"indexname": "idx_atoms_embedding_ivfflat", "indexdef": FROZEN_ANN}]
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 0
    reindex_calls = [call for call in store.calls if call[1].startswith("REINDEX INDEX")]
    assert len(reindex_calls) == 1
    assert "idx_atoms_embedding_ivfflat" in reindex_calls[0][1]
    assert "CONCURRENTLY" not in reindex_calls[0][1]


async def test_unknown_ann_index_fails_closed():
    store = _store_with_history()
    store.pg_indexes = [
        {
            "indexname": "atoms_embedding_hnsw",
            "indexdef": "CREATE INDEX atoms_embedding_hnsw ON noesis_core.atoms USING hnsw (embedding)",
        }
    ]
    assert await _run(store) == 3
    assert store.profile["status"] == "ready"


async def test_multiple_ann_indexes_fail_closed():
    store = _store_with_history()
    store.pg_indexes = [
        {"indexname": "idx_atoms_embedding_ivfflat", "indexdef": FROZEN_ANN},
        {
            "indexname": "xtra",
            "indexdef": "CREATE INDEX xtra ON noesis_core.atoms USING hnsw (embedding)",
        },
    ]
    assert await _run(store) == 3


# ---------------------------------------------------------------------------
# §6.13/§7 fail-closed cutover
# ---------------------------------------------------------------------------

async def test_encoding_failure_leaves_live_data_and_profile_rebuilding():
    store = _store_with_history()
    failing = FakeEmbeddingClient(identity_failures={("张三", "E"): "timeout"})
    assert await _run(store, failing, spec=_spec(revision="bge-m3-1024-v2")) == 6
    assert store.profile["status"] == "rebuilding"
    assert store.anchors[105]["total_count"] == 7
    assert store.atoms[("张三", "E")]["embedding"] == fake_identity_vector("张三", "E")


async def test_context_encoding_failure_leaves_live_data_untouched():
    store = _store_with_history()
    failing = FakeEmbeddingClient(context_failures={"张三 修复 服务器": "timeout"})
    assert await _run(store, failing, spec=_spec(revision="bge-m3-1024-v2")) == 6
    assert store.profile["status"] == "rebuilding"
    assert store.anchors[105]["total_count"] == 7


async def test_anchor_update_failure_rolls_back_atoms_too():
    store = _store_with_history()
    store.fail_after = {"FROM pg_temp.noesis_embedding_anchor_stage": 0}
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 9
    assert store.profile["status"] == "rebuilding"
    # The atom cutover in the same transaction was rolled back.
    assert store.atoms[("张三", "E")]["embedding"] == fake_identity_vector("张三", "E")
    assert store.anchors[105]["total_count"] == 7


async def test_reindex_failure_rolls_back_the_whole_generation():
    store = _store_with_history()
    store.pg_indexes = [{"indexname": "idx_atoms_embedding_ivfflat", "indexdef": FROZEN_ANN}]
    store.fail_after = {"REINDEX INDEX": 0}
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 9
    assert store.profile["status"] == "rebuilding"
    assert store.anchors[105]["total_count"] == 7


async def test_profile_ready_update_failure_rolls_back_the_whole_generation():
    store = _store_with_history()
    store.fail_after = {"SET model_name": 0}
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 9
    assert store.profile["status"] == "rebuilding"
    assert store.atoms[("张三", "E")]["embedding"] == fake_identity_vector("张三", "E")


async def test_rerun_after_interrupted_rebuild():
    store = _store_with_history()
    store.profile["status"] = "rebuilding"
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 0
    assert store.profile["status"] == "ready"
    assert store.anchors[105]["total_count"] == 3


# ---------------------------------------------------------------------------
# §6.14 --repair-null-only
# ---------------------------------------------------------------------------

async def test_repair_null_only_fills_nulls_without_touching_anchors_index_or_profile():
    store = _store_with_history()
    store.atoms[("张三", "E")]["embedding"] = None
    store.pg_indexes = [{"indexname": "idx_atoms_embedding_ivfflat", "indexdef": FROZEN_ANN}]

    assert await _run(store, repair_null_only=True) == 0

    assert store.atoms[("张三", "E")]["embedding"] == fake_identity_vector("张三", "E")
    assert store.atoms[("李四", "E")]["embedding"] == fake_identity_vector("李四", "E")
    assert store.anchors[105]["total_count"] == 7  # anchors untouched
    assert store.profile["status"] == "ready"
    assert store.profile["model_revision"] == "bge-m3-1024-v1"
    assert not any(call[1].startswith("REINDEX INDEX") for call in store.calls)
    assert not any("noesis_embedding_anchor_stage" in call[1] for call in store.calls)


async def test_repair_null_only_never_overwrites_existing_vector():
    store = _store_with_history()
    store.atoms[("张三", "E")]["embedding"] = (0.0,) * 1024
    assert await _run(store, repair_null_only=True) == 0
    assert store.atoms[("张三", "E")]["embedding"] == (0.0,) * 1024


async def test_repair_null_only_requires_matching_ready_profile():
    store = _store_with_history()
    store.profile["status"] = "rebuilding"
    assert await _run(store, repair_null_only=True) == 5

    other = _store_with_history()
    other.profile["model_revision"] = "other-r0"
    assert await _run(other, repair_null_only=True) == 5


async def test_repair_null_only_does_not_scan_events():
    store = _store_with_history()
    store.atoms[("张三", "E")]["embedding"] = None
    assert await _run(store, repair_null_only=True) == 0
    assert not any("event_id > $1" in call[1] for call in store.calls)


# ---------------------------------------------------------------------------
# CLI wiring (§6.1)
# ---------------------------------------------------------------------------

def test_cli_load_config_uses_hindsight_config(monkeypatch):
    import hindsight_api.config as config_module
    import hindsight_api.engine.retain.noesis_embedding_rebuild as rebuild

    sentinel = object()
    dotenv_loaded = False

    def load_dotenv():
        nonlocal dotenv_loaded
        dotenv_loaded = True

    monkeypatch.setattr(config_module, "load_dotenv_for_entrypoint", load_dotenv)
    monkeypatch.setattr(config_module.HindsightConfig, "from_env", classmethod(lambda cls: sentinel))

    assert rebuild._load_config() is sentinel
    assert dotenv_loaded


def test_cli_repair_and_force_are_mutually_exclusive():
    import hindsight_api.engine.retain.noesis_embedding_rebuild as rebuild

    try:
        rebuild.main(["--repair-null-only", "--force-full"])
    except SystemExit as error:
        assert error.code != 0
    else:  # pragma: no cover - argparse must reject the combination
        raise AssertionError("mutually exclusive flags were accepted")
