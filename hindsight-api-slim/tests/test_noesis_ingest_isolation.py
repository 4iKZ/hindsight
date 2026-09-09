"""R02-04 failure isolation + R02-05 observed_at stability tests.

No network, no real DB: everything goes through the in-memory FakeStore and
injected fakes. All module-global monkeypatching uses pytest monkeypatch so
state is restored automatically (R02-10 hygiene).
"""

from __future__ import annotations

import asyncio
import datetime

import pytest

import hindsight_api.engine.retain.noesis_ingest as noesis_ingest
from hindsight_api.engine.retain.noesis_ingest import ingest_noesis_batch
from tests.noesis_fakes import (
    CONTENT,
    FakeExtractOnceFactory,
    FakeIdentityClient,
    FakePool,
    FakeStore,
    golden_fact_recursive,
    identity_factory_for,
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
        identity_client_factory=kwargs.pop("identity_client_factory", identity_factory_for(FakeIdentityClient())),
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
    await ingest_noesis_batch(
        contents, "bank-1", noesis_config(), llm_config=llm_config(),
        extract_once_factory=FakeExtractOnceFactory(), pool_factory=flaky_factory,
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
        identity_client_factory=identity_factory_for(FakeIdentityClient()),
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
            identity_client_factory=identity_factory_for(FakeIdentityClient()),
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
        identity_client_factory=identity_factory_for(FakeIdentityClient()),
    )
    # One component succeeded, one failed and rolled back.
    assert len(store.events) == 1
    assert store.rollbacks >= 1
    assert len(store.alerts_by_code("event_ingest_failed")) >= 1
