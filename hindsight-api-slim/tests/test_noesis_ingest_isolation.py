"""R02-04 failure isolation + R02-05 observed_at stability tests.

No network, no real DB: everything goes through the in-memory FakeStore and
injected fakes. All module-global monkeypatching uses pytest monkeypatch so
state is restored automatically (R02-10 hygiene).
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json

import pytest

import hindsight_api.engine.retain.noesis_ingest as noesis_ingest
from hindsight_api.engine.retain.noesis_ingest import ingest_noesis_batch
from tests.noesis_fakes import (
    CONTENT,
    OBSERVED_AT,
    FakeEmbeddingClient,
    FakeExtractOnceFactory,
    FakePool,
    FakeStore,
    embedding_factory_for,
    fake_context_vector,
    fake_identity_vector,
    golden_fact_recursive,
    golden_fact_time,
    llm_config,
    noesis_config,
    outcome,
    pool_factory_for,
)


def _contents(event_date=None, document_id="doc-1"):
    item = {"content": CONTENT, "document_id": document_id}
    if event_date is not None:
        item["event_date"] = event_date
    return item


async def _run(store, contents, monkeypatch, *, components=None, **kwargs):
    def fake_extract(text, *, extract_once):
        return outcome(list(components or []))

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", fake_extract)
    return await ingest_noesis_batch(
        contents,
        "bank-1",
        noesis_config(),
        llm_config=llm_config(),
        extract_once_factory=FakeExtractOnceFactory(),
        pool_factory=kwargs.pop("pool_factory", pool_factory_for(store)),
        embedding_client_factory=kwargs.pop("embedding_client_factory", embedding_factory_for(FakeEmbeddingClient())),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# R02-05 observed_at stability
# ---------------------------------------------------------------------------

async def test_missing_time_reused_across_retries(monkeypatch):
    """Same input retried twice must reuse the queue-persisted observed_at.
    04A: events have no replay key, so the retry lands a second event by
    contract — the stability being asserted is the timestamp, not the count."""
    store = FakeStore()
    queued_item = _contents()
    queued_item["_noesis_observed_at"] = "2026-09-05T02:00:00Z"
    await _run(store, [queued_item], monkeypatch, components=[golden_fact_recursive()], operation_id="op-1")
    first_observed = next(iter(store.events.values()))["data"]["observed_at"]

    # retry the same input
    await _run(store, [dict(queued_item)], monkeypatch, components=[golden_fact_recursive()], operation_id="op-1")
    assert len(store.events) == 2  # duplicate input is a new event (04A contract)
    observed = {event["data"]["observed_at"] for event in store.events.values()}
    assert observed == {first_observed}


def test_no_unbounded_observed_at_process_cache_exists():
    assert not hasattr(noesis_ingest, "_observed_at_cache")


async def test_different_operations_same_content_distinct(monkeypatch):
    """Two genuinely different operations with identical text must not merge."""
    store = FakeStore()
    await _run(store, [_contents(document_id="doc-a")], monkeypatch, components=[golden_fact_recursive()], operation_id="op-a")
    await _run(store, [_contents(document_id="doc-b")], monkeypatch, components=[golden_fact_recursive()], operation_id="op-b")
    assert len(store.events) == 2
    observed = {e["data"]["observed_at"] for e in store.events.values()}
    assert len(observed) == 2, "different operations must use different captured times"
    # both are real events; the shared typed literals stay one atom row each
    assert len(store.atoms) == 4


async def test_explicit_time_not_overridden_by_cache(monkeypatch):
    """An explicit event_date always wins and is not cached."""
    store = FakeStore()
    ts = datetime.datetime(2026, 9, 5, 2, 0, 0, tzinfo=datetime.timezone.utc)
    await _run(store, [_contents(event_date=ts)], monkeypatch, components=[golden_fact_recursive()])
    assert next(iter(store.events.values()))["data"]["observed_at"] == "2026-09-05T02:00:00Z"


# ---------------------------------------------------------------------------
# R02-04 three-layer failure isolation
# ---------------------------------------------------------------------------

async def test_pool_failure_on_item1_does_not_stop_item2(monkeypatch):
    """Item 1's pool acquire fails; item 2 is still processed (R02-04)."""
    store = FakeStore()
    calls = {"n": 0}  # call 1 = preflight, call 2 = item1 acquire, call 3 = item1 alert, call 4 = item2

    async def flaky_factory(config):
        calls["n"] += 1
        if calls["n"] in (2, 3):  # item1 pool acquire + its alert write both fail
            raise RuntimeError("pool down")
        return FakePool(store)

    def fake_extract(text, *, extract_once):
        return outcome([golden_fact_recursive()])

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", fake_extract)
    contents = [_contents(document_id="doc-a"), _contents(document_id="doc-b")]
    # item1 fails at its pool acquire; item2 (call 4) succeeds and commits.
    # A fake bge client keeps this test about pool isolation — the requirement
    # 05 context-hard-precondition behavior has its own dedicated tests.
    await ingest_noesis_batch(
        contents, "bank-1", noesis_config(), llm_config=llm_config(),
        extract_once_factory=FakeExtractOnceFactory(), pool_factory=flaky_factory,
        embedding_client_factory=embedding_factory_for(FakeEmbeddingClient()),
    )
    assert calls["n"] >= 4, "item2 must still be attempted"
    assert len(store.events) == 1, "only item2's event should land"


async def test_unexpected_outcome_type_isolated(monkeypatch):
    """HE returning a non-ExtractionOutcome type must not kill the batch."""
    store = FakeStore()

    def weird_extract(text, *, extract_once):
        return {"components": [], "alerts": []}  # wrong type

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", weird_extract)
    # two items, both bad -> no events, no crash
    await ingest_noesis_batch(
        [_contents(document_id="doc-a"), _contents(document_id="doc-b")],
        "bank-1", noesis_config(), llm_config=llm_config(),
        extract_once_factory=FakeExtractOnceFactory(), pool_factory=pool_factory_for(store),
        embedding_client_factory=embedding_factory_for(FakeEmbeddingClient()),
    )
    assert store.events == {}


async def test_cancellation_propagates(monkeypatch):
    """asyncio.CancelledError must not be swallowed by the isolation logic."""
    def cancelled_extract(text, *, extract_once):
        raise asyncio.CancelledError()

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", cancelled_extract)
    with pytest.raises(asyncio.CancelledError):
        await ingest_noesis_batch(
            [_contents()], "bank-1", noesis_config(), llm_config=llm_config(),
            extract_once_factory=FakeExtractOnceFactory(), pool_factory=pool_factory_for(FakeStore()),
            embedding_client_factory=embedding_factory_for(FakeEmbeddingClient()),
        )


async def test_component1_rollback_component2_commits(monkeypatch):
    """A failing component rolls back while a later component of the same item
    commits independently."""
    store = FakeStore()

    def fake_extract(text, *, extract_once):
        return outcome([golden_fact_recursive(), golden_fact_recursive()])

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", fake_extract)

    # Fail exactly once, at the first component's very first SQL statement
    # (its event insert). The rollback must leave the second component able to
    # commit through its own transaction.
    calls = {"n": 0}

    async def burst_then_ok():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected event-insert failure")

    store.interleave = burst_then_ok
    await ingest_noesis_batch(
        [_contents()], "bank-1", noesis_config(), llm_config=llm_config(),
        extract_once_factory=FakeExtractOnceFactory(), pool_factory=pool_factory_for(store),
        embedding_client_factory=embedding_factory_for(FakeEmbeddingClient()),
    )
    # One component succeeded, one failed and rolled back.
    assert len(store.events) == 1
    assert store.rollbacks >= 1
    assert len(store.alerts_by_code("event_ingest_failed")) >= 1


# ---------------------------------------------------------------------------
# Requirement 05 §13.5: context/anchor failure isolation on the ingest chain
# ---------------------------------------------------------------------------

_GOLDEN_LITERALS = [("小明", "E"), ("没写", "P"), ("作业", "E"), ("揍", "P")]


def _event_atom_rows(store):
    return [args for kind, sql, args in store.calls if kind == "execute" and ".event_atoms" in sql]


async def test_identity_failure_context_success_fact_lands(monkeypatch):
    """§13.5 #6: Context succeeds, every Identity call fails (non-transport)
    → the fact still lands with NULL atom embeddings and the requirement-03
    identity_vector_failed degradation alert; anchors are still routed."""
    store = FakeStore()
    failing = FakeEmbeddingClient(identity_failures={literal: "http_4xx" for literal in _GOLDEN_LITERALS})
    await _run(
        store,
        [_contents(event_date=OBSERVED_AT)],
        monkeypatch,
        components=[golden_fact_recursive()],
        embedding_client_factory=embedding_factory_for(failing),
    )

    assert failing.context_calls  # context vectors were prepared
    assert len(store.events) == 1  # the fact still lands
    for text, atom_type in _GOLDEN_LITERALS:
        assert store.embedding_of(text, atom_type) is None  # degradable identity
    alerts = store.alerts_by_code("identity_vector_failed")
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "warning"
    assert alerts[0]["event_id"] is not None  # post-commit aggregate alert
    # anchors were still routed: every row carries a concrete anchor_id
    rows = _event_atom_rows(store)
    assert len(rows) == 5 and all(row[5] is not None for row in rows)
    assert len(store.anchors_rows()) == 4


async def test_context_health_unavailable_drops_components_isolated(monkeypatch):
    """§13.5 #7: /health unreachable → zero fact state per component, one
    anchor_context_service_unavailable alert each, and the remaining
    components of the same item are still processed."""
    store = FakeStore()
    unavailable = FakeEmbeddingClient(health="unavailable")

    def fake_extract(text, *, extract_once):
        return outcome([golden_fact_recursive(), golden_fact_recursive()])

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", fake_extract)
    await ingest_noesis_batch(
        [_contents(event_date=OBSERVED_AT)], "bank-1", noesis_config(), llm_config=llm_config(),
        extract_once_factory=FakeExtractOnceFactory(), pool_factory=pool_factory_for(store),
        embedding_client_factory=embedding_factory_for(unavailable),
    )

    assert store.events == {}
    assert store.atoms == {}
    assert store.event_atoms == []
    assert store.anchors == {}
    alerts = store.alerts_by_code("anchor_context_service_unavailable")
    assert len(alerts) == 2  # one per component — the component loop continued
    assert all(alert["severity"] == "error" for alert in alerts)
    assert {alert["details"]["component_index"] for alert in alerts} == {0, 1}
    assert alerts[0]["details"]["exception_type"] == "EmbeddingServiceUnavailable"


async def test_context_vector_failure_drops_component_others_land(monkeypatch):
    """§13.5 #7: a per-frame embed failure (non-transport) drops only its own
    component with an anchor_context_vector_failed alert; the next component
    of the same item still lands with routed anchors."""
    store = FakeStore()
    failing = FakeEmbeddingClient(context_failures={"没写 作业": "http_4xx"})
    await _run(
        store,
        [_contents(event_date=OBSERVED_AT)],
        monkeypatch,
        components=[golden_fact_recursive(), golden_fact_time()],
        embedding_client_factory=embedding_factory_for(failing),
    )

    assert len(store.events) == 1  # only the second component landed
    alerts = store.alerts_by_code("anchor_context_vector_failed")
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "error"
    assert alerts[0]["details"]["component_index"] == 0
    assert alerts[0]["details"]["error_kind"] == "http_4xx"
    assert alerts[0]["details"]["exception_type"] == "EmbeddingError"
    # the surviving component's rows carry routed anchor ids
    rows = _event_atom_rows(store)
    assert len(rows) == 5 and all(row[5] is not None for row in rows)
    assert len(store.anchors_rows()) == 5


# ---------------------------------------------------------------------------
# Requirement 05A §11.1: shared embedding-space profile gate
# ---------------------------------------------------------------------------


def _seed_profile(model="bge-m3", revision="bge-m3-1024-v1", dimension=1024, status="ready"):
    return {
        "embedding_kind": "identity",
        "model_name": model,
        "model_revision": revision,
        "dimension": dimension,
        "status": status,
    }


def _profile_alerts(store):
    return store.alerts_by_code("embedding_profile_unavailable")


def _assert_gate_rejection(store, client, *, reason):
    """05A §5.3 gate-refusal contract: zero facts, zero anchors, zero bge
    HTTP (§11.1 #9), exactly one fixed sanitized alert, native chain intact."""
    assert store.events == {}
    assert store.event_atoms == []
    # §11.1 #7/#8 seed pre-existing atoms/anchors, so the refusal contract
    # here is "zero atom/anchor WRITES" — the gate must never claim, upsert,
    # or touch them — not "the tables are empty".
    assert not any(
        "INSERT INTO" in sql and (".atoms" in sql or ".anchors" in sql) for _, sql, _ in store.calls
    )
    assert client.identity_calls == []  # §11.1 #9: zero /normalize/entity|predicate
    assert client.context_calls == []  # §11.1 #9: zero /normalize/sentence
    assert store.alerts_by_code("event_ingest_failed") == []  # no generic failure path
    alerts = _profile_alerts(store)
    assert len(alerts) == 1  # §11.1 #10: the fixed alert, exactly once
    alert = alerts[0]
    assert alert["stage"] == "embedding_profile"
    assert alert["alert_code"] == "embedding_profile_unavailable"
    assert alert["severity"] == "error"
    assert alert["message"] == "noesis embedding generation is unavailable; component dropped"
    assert alert["details"]["reason"] == reason


# §11.1 #1
async def test_gate_ready_exact_match_allows_ingest(monkeypatch):
    store = FakeStore()
    store.profile = _seed_profile()
    await _run(store, [_contents(event_date=OBSERVED_AT)], monkeypatch, components=[golden_fact_recursive()])

    assert len(store.events) == 1  # allowed: triple matches and status is ready
    assert len(store.anchors_rows()) == 4
    assert _profile_alerts(store) == []


# §11.1 #2 (+ #9 zero-HTTP, #10 fixed alert folded into _assert_gate_rejection)
async def test_gate_model_mismatch_rejects_before_any_bge_http(monkeypatch):
    store = FakeStore()
    store.profile = _seed_profile(model="other-model", revision="other-r0")
    client = FakeEmbeddingClient()
    await _run(
        store,
        [_contents(event_date=OBSERVED_AT)],
        monkeypatch,
        components=[golden_fact_recursive()],
        embedding_client_factory=embedding_factory_for(client),
    )

    _assert_gate_rejection(store, client, reason="generation_mismatch")
    details = _profile_alerts(store)[0]["details"]
    assert details["db_profile"]["model_name"] == "other-model"
    assert details["config_profile"] == {
        "model_name": "bge-m3",
        "model_revision": "bge-m3-1024-v1",
        "dimension": 1024,
    }


# §11.1 #3
async def test_gate_revision_mismatch_rejects(monkeypatch):
    store = FakeStore()
    store.profile = _seed_profile(revision="bge-m3-1024-v2")
    client = FakeEmbeddingClient()
    await _run(
        store,
        [_contents(event_date=OBSERVED_AT)],
        monkeypatch,
        components=[golden_fact_recursive()],
        embedding_client_factory=embedding_factory_for(client),
    )

    _assert_gate_rejection(store, client, reason="generation_mismatch")


# §11.1 #4
async def test_gate_dimension_mismatch_rejects(monkeypatch):
    store = FakeStore()
    store.profile = _seed_profile(dimension=768)
    client = FakeEmbeddingClient()
    await _run(
        store,
        [_contents(event_date=OBSERVED_AT)],
        monkeypatch,
        components=[golden_fact_recursive()],
        embedding_client_factory=embedding_factory_for(client),
    )

    _assert_gate_rejection(store, client, reason="generation_mismatch")


# §11.1 #5
async def test_gate_rebuilding_rejects_even_with_matching_triple(monkeypatch):
    """status=rebuilding wins over a matching triple: a rebuild in progress
    must fence every new component (05A §5.2)."""
    store = FakeStore()
    store.profile = _seed_profile(status="rebuilding")
    client = FakeEmbeddingClient()
    await _run(
        store,
        [_contents(event_date=OBSERVED_AT)],
        monkeypatch,
        components=[golden_fact_recursive()],
        embedding_client_factory=embedding_factory_for(client),
    )

    _assert_gate_rejection(store, client, reason="rebuilding")


# §11.1 #6
async def test_gate_missing_empty_store_claims_and_ingests(monkeypatch):
    """Missing profile + zero E/P vectors AND zero anchors → safe claim; the
    fact then lands normally (05A §2.3.2)."""
    store = FakeStore()
    await _run(store, [_contents(event_date=OBSERVED_AT)], monkeypatch, components=[golden_fact_recursive()])

    assert len(store.events) == 1
    assert store.profile == _seed_profile()
    assert _profile_alerts(store) == []


# §11.1 #7
async def test_gate_missing_with_existing_ep_vector_refuses(monkeypatch):
    store = FakeStore()
    store.atoms[("旧词", "E")] = {"atom_id": 42, "embedding": fake_identity_vector("旧词", "E")}
    client = FakeEmbeddingClient()
    await _run(
        store,
        [_contents(event_date=OBSERVED_AT)],
        monkeypatch,
        components=[golden_fact_recursive()],
        embedding_client_factory=embedding_factory_for(client),
    )

    _assert_gate_rejection(store, client, reason="missing_with_existing_data")
    assert store.profile is None  # never claims over existing vectors
    assert "db_profile" not in _profile_alerts(store)[0]["details"]


# §11.1 #8
async def test_gate_missing_with_existing_anchor_refuses(monkeypatch):
    store = FakeStore()
    store.seed_anchor(501, fake_context_vector("旧上下文"))
    client = FakeEmbeddingClient()
    await _run(
        store,
        [_contents(event_date=OBSERVED_AT)],
        monkeypatch,
        components=[golden_fact_recursive()],
        embedding_client_factory=embedding_factory_for(client),
    )

    _assert_gate_rejection(store, client, reason="missing_with_existing_data")
    assert store.profile is None  # never guesses a generation over anchors
    assert len(store.anchors_rows()) == 1  # the pre-existing anchor untouched


# §11.1 #10: the details whitelist, verbatim (05A §5.3)
async def test_gate_rejection_alert_details_whitelist_sanitized(monkeypatch):
    store = FakeStore()
    store.profile = _seed_profile(model="other-model", revision="other-r0", dimension=768)
    await _run(store, [_contents(event_date=OBSERVED_AT)], monkeypatch, components=[golden_fact_recursive()])

    alert = _profile_alerts(store)[0]
    details = alert["details"]
    assert set(details) == {"component_index", "reason", "db_profile", "config_profile", "content_sha256"}
    assert details["component_index"] == 0
    assert details["reason"] == "generation_mismatch"
    assert set(details["db_profile"]) == {"model_name", "model_revision", "dimension", "status"}
    assert set(details["config_profile"]) == {"model_name", "model_revision", "dimension"}
    assert details["content_sha256"] == hashlib.sha256(CONTENT.encode("utf-8")).hexdigest()
    # sanitized: no source text, atom text, vectors, or embedding endpoints
    blob = json.dumps(store.alerts, ensure_ascii=False)
    assert CONTENT not in blob
    assert "小明" not in blob and "妈妈" not in blob
    assert "identity.test" not in blob
    assert "Bearer" not in blob and "api_key" not in blob


# §11.1 #11
async def test_gate_rejection_each_component_processed_independently(monkeypatch):
    """Both components of one item are refused by the gate, each with its own
    alert — the component loop never aborts on the first rejection."""
    store = FakeStore()
    store.profile = _seed_profile(model="other-model")
    await _run(
        store,
        [_contents(event_date=OBSERVED_AT)],
        monkeypatch,
        components=[golden_fact_recursive(), golden_fact_time()],
    )

    alerts = _profile_alerts(store)
    assert len(alerts) == 2
    assert {alert["details"]["component_index"] for alert in alerts} == {0, 1}
    assert store.events == {}


# §11.1 #12
async def test_gate_rejection_keeps_native_chain_untouched(monkeypatch):
    """The rejected component only writes its alert; the batch returns
    normally, and once the profile is repaired the next item lands — the
    native Hindsight retain chain is never blocked by the gate."""
    store = FakeStore()
    store.profile = _seed_profile(model="other-model")
    await _run(store, [_contents(document_id="doc-a")], monkeypatch, components=[golden_fact_recursive()])
    assert len(_profile_alerts(store)) == 1

    store.profile = _seed_profile()  # operator repair: ready again
    await _run(store, [_contents(document_id="doc-b")], monkeypatch, components=[golden_fact_recursive()])
    assert len(store.events) == 1  # facts flow again immediately
    assert len(_profile_alerts(store)) == 1  # no new gate alert
