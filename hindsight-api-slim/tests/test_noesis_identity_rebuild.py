"""Noesis identity rebuild (requirement 03 §15.5) offline tests.

The rebuild state machine runs against the injected FakeStore + FakeIdentityClient;
no real database, no real network. Each test drives ``run_identity_rebuild``
directly and asserts the state machine's invariants (lock, rebuilding status,
no live mutation on failure, atomic switch, repair-null-only guards).
"""

from __future__ import annotations

from hindsight_api.engine.retain.noesis_identity_rebuild import (
    BuildSpec,
    run_identity_rebuild,
)
from tests.noesis_fakes import (
    FakeIdentityClient,
    FakePool,
    FakeStore,
    fake_identity_vector,
)


def test_cli_load_config_uses_hindsight_config(monkeypatch):
    """The documented ``python -m`` entry point must load the real config type."""
    import hindsight_api.config as config_module
    import hindsight_api.engine.retain.noesis_identity_rebuild as rebuild

    sentinel = object()
    dotenv_loaded = False

    def load_dotenv():
        nonlocal dotenv_loaded
        dotenv_loaded = True

    monkeypatch.setattr(config_module, "load_dotenv_for_entrypoint", load_dotenv)
    monkeypatch.setattr(
        config_module.HindsightConfig,
        "from_env",
        classmethod(lambda cls: sentinel),
    )

    assert rebuild._load_config() is sentinel
    assert dotenv_loaded


def _spec(model: str = "bge-m3", revision: str = "bge-m3-1024-v1", dimension: int = 1024) -> BuildSpec:
    return BuildSpec(model=model, revision=revision, dimension=dimension)


def _store_with_filled_atoms(profile=True, n_ep=4) -> FakeStore:
    store = FakeStore()
    if profile:
        store.profile = {
            "embedding_kind": "identity",
            "model_name": "bge-m3",
            "model_revision": "bge-m3-1024-v1",
            "dimension": 1024,
            "status": "ready",
        }
    literals = [("小明", "E"), ("作业", "E"), ("没写", "P"), ("揍", "P")]
    for index, (text, atom_type) in enumerate(literals[:n_ep]):
        store.atoms[(text, atom_type)] = {
            "atom_id": 500 + index,
            "embedding": fake_identity_vector(text, atom_type),
        }
    # A G atom must never be touched/encoded.
    store.atoms[("周末计划", "G")] = {"atom_id": 900, "embedding": None}
    return store


async def _run(store, client=None, *, spec=None, repair_null_only=False, schema="noesis_core"):
    return await run_identity_rebuild(
        pool=FakePool(store),
        schema=schema,
        client=client or FakeIdentityClient(),
        spec=spec or _spec(),
        repair_null_only=repair_null_only,
    )


# ---------------------------------------------------------------------------
# §15.5 lock / concurrency
# ---------------------------------------------------------------------------

async def test_second_rebuild_refused_when_lock_held():
    store = _store_with_filled_atoms()
    store.advisory_lock_available = False  # a rebuild is already running
    assert await _run(store) == 2


# ---------------------------------------------------------------------------
# §15.5 safe failure: live vectors untouched, profile stays rebuilding
# ---------------------------------------------------------------------------

async def test_any_encoding_failure_leaves_live_vectors_and_profile_rebuilding():
    store = _store_with_filled_atoms()
    failing = FakeIdentityClient(failures={("小明", "E"): "timeout"})
    # A different revision forces an actual recompute (not the no-op branch).
    spec = _spec(revision="bge-m3-1024-v2")
    assert await _run(store, failing, spec=spec) == 6

    # live vectors untouched
    for (text, atom_type), atom in store.atoms.items():
        if atom_type == "E" and text == "小明":
            assert atom["embedding"] is not None  # still the original vector
    # profile stays rebuilding (the failure happened before the final commit)
    assert store.profile["status"] == "rebuilding"


async def test_interrupted_run_is_rerunnable_and_idempotent():
    store = _store_with_filled_atoms()
    # First a full run to a NEW generation succeeds and switches profile.
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 0
    assert store.profile["model_revision"] == "bge-m3-1024-v2"
    assert store.profile["status"] == "ready"
    for (text, atom_type), atom in store.atoms.items():
        if atom_type in ("E", "P"):
            assert atom["embedding"] is not None
    # Re-running to the same generation is a no-op.
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 0
    assert store.profile["status"] == "ready"


async def test_resuming_an_interrupted_rebuild_is_allowed():
    store = _store_with_filled_atoms()
    store.profile["status"] = "rebuilding"  # a prior run died mid-encode
    client = FakeIdentityClient()
    assert await _run(store, client) == 0
    assert store.profile["status"] == "ready"


# ---------------------------------------------------------------------------
# §15.5 atomic switch + G untouched
# ---------------------------------------------------------------------------

async def test_full_switch_updates_all_ep_and_keeps_g_null():
    store = _store_with_filled_atoms()
    new_spec = _spec(model="bge-m3", revision="bge-m3-1024-v2", dimension=1024)
    assert await _run(store, spec=new_spec) == 0

    assert store.profile["model_revision"] == "bge-m3-1024-v2"
    assert store.profile["status"] == "ready"
    for (text, atom_type), atom in store.atoms.items():
        if atom_type in ("E", "P"):
            assert atom["embedding"] == fake_identity_vector(text, atom_type)
        elif atom_type == "G":
            assert atom["embedding"] is None  # G never gets a vector


async def test_full_switch_includes_atom_created_while_rebuilding():
    """Facts keep flowing in rebuilding, so the final generation must catch late E/P atoms."""
    store = _store_with_filled_atoms()
    client = FakeIdentityClient()
    original_embed = client.embed
    inserted = False

    async def embed_and_insert_late_atom(text, atom_type):
        nonlocal inserted
        vector = await original_embed(text, atom_type)
        if not inserted:
            inserted = True
            store.atoms[("晚到意元", "E")] = {
                "atom_id": 999,
                "embedding": None,
            }
        return vector

    client.embed = embed_and_insert_late_atom
    assert await _run(store, client, spec=_spec(revision="bge-m3-1024-v2")) == 0
    assert ("晚到意元", "E") in client.embed_calls
    assert store.atoms[("晚到意元", "E")]["embedding"] == fake_identity_vector("晚到意元", "E")


async def test_full_switch_uses_database_staging_and_set_based_cutover():
    store = _store_with_filled_atoms()
    assert await _run(store, spec=_spec(revision="bge-m3-1024-v2")) == 0

    sql = "\n".join(call[1] for call in store.calls)
    assert "CREATE TEMP TABLE" in sql
    assert "FROM pg_temp.noesis_identity_rebuild_stage" in sql
    assert "LOCK TABLE noesis_core.atoms" in sql


async def test_encoding_happens_before_any_live_update():
    store = _store_with_filled_atoms()
    client = FakeIdentityClient()

    async def spy_embed(text, atom_type):
        # Assert no live atom has been mutated yet at every encode call.
        for (t, ty), atom in store.atoms.items():
            if ty in ("E", "P"):
                assert atom["embedding"] is not None  # original, untouched
        return fake_identity_vector(text, atom_type)

    client.embed = spy_embed
    assert await _run(store, client, spec=_spec(revision="bge-m3-1024-v2")) == 0


# ---------------------------------------------------------------------------
# §15.5 --repair-null-only
# ---------------------------------------------------------------------------

async def test_repair_null_only_fills_nulls_without_touching_existing():
    store = _store_with_filled_atoms()
    # One E atom starts as NULL (was never embedded by an earlier bge outage).
    store.atoms[("小明", "E")]["embedding"] = None
    assert await _run(store, repair_null_only=True) == 0

    assert store.atoms[("小明", "E")]["embedding"] == fake_identity_vector("小明", "E")
    # Existing vector untouched.
    assert store.atoms[("作业", "E")]["embedding"] == fake_identity_vector("作业", "E")
    # Profile unchanged.
    assert store.profile["model_revision"] == "bge-m3-1024-v1"
    assert store.profile["status"] == "ready"


async def test_repair_null_only_never_overwrites_existing_vector():
    store = _store_with_filled_atoms()
    # Bump the existing E atom's vector to a sentinel that repair must preserve.
    store.atoms[("小明", "E")]["embedding"] = (0.0,) * 1024
    client = FakeIdentityClient()
    assert await _run(store, client, repair_null_only=True) == 0
    assert store.atoms[("小明", "E")]["embedding"] == (0.0,) * 1024  # untouched


async def test_repair_null_only_requires_matching_ready_profile():
    store = _store_with_filled_atoms()
    store.profile["status"] = "rebuilding"
    assert await _run(store, repair_null_only=True) == 5

    store2 = _store_with_filled_atoms()
    store2.profile["model_revision"] = "other-r0"
    assert await _run(store2, repair_null_only=True) == 5


async def test_repair_encoding_failure_leaves_live_atoms_untouched():
    store = _store_with_filled_atoms()
    # Two NULL E/P atoms await repair; the second one's encoding fails.
    store.atoms[("小明", "E")]["embedding"] = None
    store.atoms[("揍", "P")]["embedding"] = None
    failing = FakeIdentityClient(failures={("揍", "P"): "timeout"})
    assert await _run(store, failing, repair_null_only=True) == 6

    # Atomic repair: nothing was written, not even the successfully encoded one.
    assert store.atoms[("小明", "E")]["embedding"] is None
    assert store.atoms[("揍", "P")]["embedding"] is None
    # Existing vectors and the profile are untouched.
    assert store.atoms[("作业", "E")]["embedding"] == fake_identity_vector("作业", "E")
    assert store.atoms[("没写", "P")]["embedding"] == fake_identity_vector("没写", "P")
    assert store.profile["model_revision"] == "bge-m3-1024-v1"
    assert store.profile["status"] == "ready"


async def test_repair_encoding_happens_before_any_live_update():
    store = _store_with_filled_atoms()
    store.atoms[("小明", "E")]["embedding"] = None
    store.atoms[("揍", "P")]["embedding"] = None
    client = FakeIdentityClient()

    async def spy_embed(text, atom_type):
        # Every encode call must happen before ANY live atom is backfilled:
        # the NULL targets must still be NULL while encoding is in progress.
        assert store.atoms[("小明", "E")]["embedding"] is None
        assert store.atoms[("揍", "P")]["embedding"] is None
        return fake_identity_vector(text, atom_type)

    client.embed = spy_embed
    assert await _run(store, client, repair_null_only=True) == 0
    assert store.atoms[("小明", "E")]["embedding"] == fake_identity_vector("小明", "E")
    assert store.atoms[("揍", "P")]["embedding"] == fake_identity_vector("揍", "P")


# ---------------------------------------------------------------------------
# §15.5 ANN index refusal
# ---------------------------------------------------------------------------

async def test_full_rebuild_refused_when_ann_index_exists():
    store = _store_with_filled_atoms()
    store.pg_indexes = [
        {
            "indexname": "atoms_embedding_hnsw",
            "indexdef": "CREATE INDEX atoms_embedding_hnsw ON noesis_core.atoms USING hnsw (embedding)",
        }
    ]
    assert await _run(store) == 3
    # Profile untouched (no rebuilding begun).
    assert store.profile["status"] == "ready"


async def test_repair_null_only_allowed_even_with_ann_index():
    store = _store_with_filled_atoms()
    store.atoms[("小明", "E")]["embedding"] = None
    store.pg_indexes = [
        {
            "indexname": "atoms_embedding_hnsw",
            "indexdef": "CREATE INDEX atoms_embedding_hnsw ON noesis_core.atoms USING hnsw (embedding)",
        }
    ]
    assert await _run(store, repair_null_only=True) == 0


# ---------------------------------------------------------------------------
# §15.5 client health failure
# ---------------------------------------------------------------------------

async def test_client_not_ready_aborts_without_mutation():
    store = _store_with_filled_atoms()
    client = FakeIdentityClient(health="config_invalid")
    assert await _run(store, client) == 4
    assert store.profile["status"] == "ready"  # unchanged
    for (text, atom_type), atom in store.atoms.items():
        if atom_type in ("E", "P"):
            assert atom["embedding"] == fake_identity_vector(text, atom_type)
