"""Noesis ingestion contract tests: batch handling, routing, isolation.

Requirement 02 §17.1 / §17.2. All extraction is a fake (module-level
``extract_noesis_components`` seam or ``extract_once_factory`` seam); all
database access goes through the in-memory FakePool. No network, no real DB.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

import hindsight_api.engine.retain.noesis_ingest as noesis_ingest
from hindsight_api.engine.retain.noesis_ingest import ingest_noesis_batch
from tests.noesis_fakes import (
    CONTENT,
    OBSERVED_AT,
    FakeExtractOnceFactory,
    FakeIdentityClient,
    FakeStore,
    golden_fact_recursive,
    golden_fact_time,
    golden_hypothesis,
    he_alert,
    identity_factory_for,
    llm_config,
    noesis_config,
    outcome,
    pool_factory_for,
)


def patch_outcomes(monkeypatch, outcomes):
    """Serve canned ExtractionOutcomes through the module-level seam."""
    calls: list[str] = []

    def fake_extract_noesis_components(text, *, extract_once):
        calls.append(text)
        result = outcomes.pop(0) if outcomes else outcome()
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", fake_extract_noesis_components)
    return calls


def content_item(content=CONTENT, event_date=OBSERVED_AT, document_id=None, **extra):
    item = {"content": content}
    if event_date is not None:
        item["event_date"] = event_date
    if document_id is not None:
        item["document_id"] = document_id
    item.update(extra)
    return item


async def run_ingest(store, contents, monkeypatch, *, config=None, llm_config_value=None, **kwargs):
    """Default seam set: fake extract_once factory + FakePool.

    Pass ``extract_once_factory=None`` explicitly to exercise the production
    factory (used by the noesis_llm_config_invalid tests)."""
    cfg = config or noesis_config()
    return await ingest_noesis_batch(
        contents,
        "bank-1",
        cfg,
        llm_config=llm_config_value,
        extract_once_factory=kwargs.pop("extract_once_factory", FakeExtractOnceFactory()),
        pool_factory=kwargs.pop("pool_factory", pool_factory_for(store)),
        # Offline + alert-quiet default; requirement-03 vector behavior is
        # covered by test_noesis_identity_vector.py.
        identity_client_factory=kwargs.pop("identity_client_factory", identity_factory_for(FakeIdentityClient())),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# §17.1 batch handling
# ---------------------------------------------------------------------------

async def test_empty_batch_skips_extraction_and_pool(monkeypatch):
    calls = patch_outcomes(monkeypatch, [])
    store = FakeStore()
    await run_ingest(store, [], monkeypatch)
    assert calls == []
    assert store.calls == []


async def test_blank_items_skipped_without_llm_or_alerts(monkeypatch):
    calls = patch_outcomes(monkeypatch, [])
    store = FakeStore()
    await run_ingest(store, [content_item(""), content_item("   \n\t ")], monkeypatch)
    assert calls == []
    assert store.alerts == []
    assert store.events == {}


async def test_every_nonempty_item_extracted_once_with_own_envelope(monkeypatch):
    calls = patch_outcomes(monkeypatch, [outcome(), outcome(), outcome()])
    store = FakeStore()
    ts_a = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    ts_b = datetime(2026, 9, 2, 9, 0, tzinfo=UTC)
    contents = [
        content_item("第一件事。", event_date=ts_a, document_id="doc-a"),
        content_item("第二件事。", event_date=ts_b),
        content_item("第三件事。", event_date=None),
    ]
    await run_ingest(store, contents, monkeypatch, operation_id="op-7")

    assert calls == ["第一件事。", "第二件事。", "第三件事。"]


async def test_item_envelope_fields_recorded_per_item(monkeypatch):
    patch_outcomes(monkeypatch, [outcome([golden_fact_recursive()]), outcome([golden_fact_recursive()])])
    store = FakeStore()
    ingested: list = []
    real_ingest_fact = noesis_ingest._ingest_fact

    async def spy_ingest_fact(*args, **kwargs):
        ingested.append(kwargs["item"])
        return await real_ingest_fact(*args, **kwargs)

    monkeypatch.setattr(noesis_ingest, "_ingest_fact", spy_ingest_fact)

    contents = [content_item("甲。"), content_item("乙。")]
    await run_ingest(store, contents, monkeypatch, operation_id="op-7")

    assert [item.item_index for item in ingested] == [0, 1]
    assert all(item.bank_id == "bank-1" for item in ingested)
    assert all(item.operation_id == "op-7" for item in ingested)
    assert all(item.source == "hindsight_retain" for item in ingested)


async def test_item_envelope_preserves_original_batch_index(monkeypatch):
    patch_outcomes(monkeypatch, [outcome([golden_fact_recursive()])])
    store = FakeStore()
    ingested: list = []
    real_ingest_fact = noesis_ingest._ingest_fact

    async def spy_ingest_fact(*args, **kwargs):
        ingested.append(kwargs["item"])
        return await real_ingest_fact(*args, **kwargs)

    monkeypatch.setattr(noesis_ingest, "_ingest_fact", spy_ingest_fact)
    await run_ingest(
        store,
        [content_item("survivor", _noesis_item_index=7)],
        monkeypatch,
        operation_id="op-original-index",
    )
    assert [item.item_index for item in ingested] == [7]


async def test_alert_identity_includes_observed_at():
    first = noesis_ingest.NoesisInputItem(
        bank_id="bank", content="same", observed_at=datetime(2026, 9, 5, 1, tzinfo=UTC),
        operation_id=None, document_id=None, item_index=0
    )
    second = noesis_ingest.NoesisInputItem(
        bank_id="bank", content="same", observed_at=datetime(2026, 9, 5, 2, tzinfo=UTC),
        operation_id=None, document_id=None, item_index=0
    )
    assert noesis_ingest.compute_alert_dedupe_key(
        item=first, stage="extract", alert_code="bad", component_index=None
    ) != noesis_ingest.compute_alert_dedupe_key(
        item=second, stage="extract", alert_code="bad", component_index=None
    )


async def test_missing_timestamp_captured_once_at_batch_boundary(monkeypatch):
    """Items without event_date share one batch-boundary now(), and their
    observed_at differs from a later batch's now()."""
    store = FakeStore()
    fixed = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
    patch_outcomes(monkeypatch, [outcome([golden_fact_recursive()]), outcome([golden_fact_recursive()])])

    ingested: list = []
    real_ingest_fact = noesis_ingest._ingest_fact

    async def spy_ingest_fact(*args, **kwargs):
        ingested.append(kwargs["item"])
        return await real_ingest_fact(*args, **kwargs)

    monkeypatch.setattr(noesis_ingest, "_ingest_fact", spy_ingest_fact)

    await run_ingest(store, [content_item(event_date=None), content_item(event_date=None)], monkeypatch, clock=lambda: fixed)
    assert len(ingested) == 2
    assert all(item.observed_at == fixed for item in ingested)


async def test_noesis_disabled_is_noop(monkeypatch):
    calls = patch_outcomes(monkeypatch, [])

    class _Boom:
        def __call__(self, llm_config):
            raise AssertionError("extract_once_factory must not run when disabled")

    store = FakeStore()
    cfg = noesis_config(noesis_enabled=False)
    await ingest_noesis_batch(
        [content_item()], "bank-1", cfg, llm_config=llm_config(),
        extract_once_factory=_Boom(), pool_factory=pool_factory_for(store),
    )
    assert calls == []
    assert store.calls == []


# ---------------------------------------------------------------------------
# §17.2 output routing
# ---------------------------------------------------------------------------

async def test_empty_components_no_events_no_alerts(monkeypatch):
    calls = patch_outcomes(monkeypatch, [outcome()])
    store = FakeStore()
    await run_ingest(store, [content_item()], monkeypatch)
    assert calls == [CONTENT]
    assert store.events == {}
    assert store.alerts == []


async def test_fact_written_to_fact_layer(monkeypatch):
    patch_outcomes(monkeypatch, [outcome([golden_fact_recursive()])])
    store = FakeStore()
    await run_ingest(store, [content_item()], monkeypatch)
    assert len(store.events) == 1
    event = next(iter(store.events.values()))
    assert event["data"]["component"]["utterance_type"] == "fact"


async def test_hypothesis_classification_pending_alert_only(monkeypatch):
    """04A §3.3: hypothesis is an observability signal only — never an event,
    never a persisted rule; the alert message must not claim persistence."""
    patch_outcomes(monkeypatch, [outcome([golden_hypothesis()])])
    store = FakeStore()
    await run_ingest(store, [content_item()], monkeypatch)

    assert store.events == {}
    assert store.atoms == {}
    assert store.event_atoms == []
    pending = store.alerts_by_code("hypothesis_classification_pending")
    assert len(pending) == 1
    alert = pending[0]
    assert alert["stage"] == "hypothesis_routing"
    assert alert["severity"] == "info"
    assert alert["event_id"] is None
    assert alert["details"]["component"]["utterance_type"] == "hypothesis"
    assert alert["details"]["bank_id"] == "bank-1"
    message = alert["message"]
    assert "not" in message and "persisted" in message  # purely observational wording


async def test_hyper_extract_alerts_mapped_with_safe_envelope(monkeypatch):
    patch_outcomes(monkeypatch, [outcome(alerts=[he_alert()])])
    store = FakeStore()
    await run_ingest(store, [content_item()], monkeypatch, operation_id="op-1")

    alerts = store.alerts_by_code("invalid_component_dropped")
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["stage"] == "hyper_extract"
    assert alert["severity"] == "warning"
    assert alert["event_id"] is None
    envelope = alert["details"]
    assert envelope["bank_id"] == "bank-1"
    assert envelope["operation_id"] == "op-1"
    assert envelope["document_id"] is None
    assert envelope["item_index"] == 0
    assert envelope["attempts"] == 1
    assert len(envelope["content_sha256"]) == 64
    # original HE alert details preserved
    assert envelope["rule"] == "closure"


async def test_extraction_failure_alert_and_isolation(monkeypatch):
    calls = patch_outcomes(monkeypatch, [outcome(), RuntimeError("llm down")])
    store = FakeStore()
    await run_ingest(store, [content_item("甲。"), content_item("乙。")], monkeypatch)

    assert calls == ["甲。", "乙。"]  # batch continued after failure
    failed = store.alerts_by_code("extraction_failed")
    assert len(failed) == 1
    assert failed[0]["severity"] == "error"
    assert failed[0]["stage"] == "hyper_extract"
    assert failed[0]["event_id"] is None


async def test_mixed_fact_hypothesis_routes_separately(monkeypatch):
    patch_outcomes(monkeypatch, [outcome([golden_fact_recursive(), golden_hypothesis()])])
    store = FakeStore()
    await run_ingest(store, [content_item()], monkeypatch)
    assert len(store.events) == 1
    assert len(store.alerts_by_code("hypothesis_classification_pending")) == 1


async def test_multiple_components_in_order(monkeypatch):
    patch_outcomes(monkeypatch, [outcome([golden_fact_time(), golden_fact_recursive()])])
    store = FakeStore()
    await run_ingest(store, [content_item()], monkeypatch)

    assert len(store.events) == 2
    component_indexes = sorted(event["data"]["component_index"] for event in store.events.values())
    assert component_indexes == [0, 1]
    first = next(event for event in store.events.values() if event["data"]["component_index"] == 0)
    assert first["data"]["component"]["tree"]["predicate"] == "买"


async def test_no_llm_config_writes_config_alert(monkeypatch):
    calls = patch_outcomes(monkeypatch, [])
    store = FakeStore()
    await run_ingest(store, [content_item()], monkeypatch, llm_config_value=None, extract_once_factory=None)

    assert calls == []
    alerts = store.alerts_by_code("noesis_llm_config_invalid")
    assert len(alerts) == 1
    assert alerts[0]["stage"] == "noesis_config"
    assert alerts[0]["severity"] == "error"


async def test_unsupported_provider_writes_config_alert(monkeypatch):
    calls = patch_outcomes(monkeypatch, [])
    store = FakeStore()
    await run_ingest(
        store, [content_item()], monkeypatch,
        llm_config_value=llm_config(provider="gemini", model="gemini-2"),
        extract_once_factory=None,
    )

    assert calls == []
    assert len(store.alerts_by_code("noesis_llm_config_invalid")) == 1


async def test_invalid_schema_identifier_skips_with_log(monkeypatch, caplog):
    calls = patch_outcomes(monkeypatch, [outcome([golden_fact_recursive()])])
    store = FakeStore()
    with caplog.at_level("ERROR"):
        await run_ingest(store, [content_item()], monkeypatch, config=noesis_config(noesis_schema="bad name"))
    assert calls == []
    assert store.events == {}
    assert any("noesis" in record.message.lower() for record in caplog.records)


async def test_alerts_and_logs_never_leak_secrets(monkeypatch, caplog):
    secret_llm = llm_config()
    secret_llm.api_key = "sk-super-secret-key"
    secret_content = "数据库密码是 hunter2，明天见面说。"
    patch_outcomes(monkeypatch, [outcome([golden_fact_recursive()]), RuntimeError("boom")])
    store = FakeStore()
    with caplog.at_level("DEBUG"):
        await run_ingest(
            store,
            [content_item(secret_content), content_item()],
            monkeypatch,
            llm_config_value=secret_llm,
        )

    alert_json = json.dumps(store.alerts, ensure_ascii=False)
    assert "sk-super-secret-key" not in alert_json
    assert "hunter2" not in alert_json
    assert secret_content not in alert_json
    for record in caplog.records:
        message = record.getMessage()
        assert "sk-super-secret-key" not in message
        assert noesis_config().noesis_database_url not in message


async def test_batch_clock_shared_for_all_components(monkeypatch):
    """One fact per item, none of the items has a timestamp: all observed_at
    values come from the single batch-boundary capture."""
    store = FakeStore()
    fixed = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
    patch_outcomes(monkeypatch, [outcome([golden_fact_recursive()])] * 3)
    await run_ingest(
        store,
        [content_item(event_date=None) for _ in range(3)],
        monkeypatch,
        clock=lambda: fixed,
    )
    observed = {event["data"]["observed_at"] for event in store.events.values()}
    assert observed == {"2026-09-05T12:00:00Z"}


async def test_item_document_id_beats_batch_document_id(monkeypatch):
    patch_outcomes(monkeypatch, [outcome([golden_fact_recursive()]), outcome([golden_fact_recursive()])])
    store = FakeStore()
    ingested: list = []
    real_ingest_fact = noesis_ingest._ingest_fact

    async def spy_ingest_fact(*args, **kwargs):
        ingested.append(kwargs["item"])
        return await real_ingest_fact(*args, **kwargs)

    monkeypatch.setattr(noesis_ingest, "_ingest_fact", spy_ingest_fact)

    contents = [content_item(document_id="item-doc"), content_item()]
    await ingest_noesis_batch(
        contents, "bank-1", noesis_config(),
        llm_config=llm_config(), document_id="batch-doc",
        extract_once_factory=FakeExtractOnceFactory(),
        pool_factory=pool_factory_for(store),
        identity_client_factory=identity_factory_for(FakeIdentityClient()),
    )
    assert [item.document_id for item in ingested] == ["item-doc", "batch-doc"]


# ---------------------------------------------------------------------------
# §14.3 / §17.7 pool lifecycle
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
async def _reset_pool_singleton():
    await noesis_ingest.close_noesis_pool()
    yield
    await noesis_ingest.close_noesis_pool()


async def test_pool_concurrent_lazy_init_once(monkeypatch):
    import asyncio

    opened: list = []

    async def fake_open(config):
        await asyncio.sleep(0.01)  # widen the race window
        opened.append(config)
        return object()

    monkeypatch.setattr(noesis_ingest, "_open_pool", fake_open)
    cfg = noesis_config()
    await asyncio.gather(*(noesis_ingest._get_pool(cfg) for _ in range(5)))
    assert len(opened) == 1


async def test_close_noesis_pool_idempotent(monkeypatch):
    closes: list[int] = []

    class _Pool:
        def is_closed(self) -> bool:
            return False

        async def close(self) -> None:
            closes.append(1)

    async def fake_open(config):
        return _Pool()

    monkeypatch.setattr(noesis_ingest, "_open_pool", fake_open)
    await noesis_ingest._get_pool(noesis_config())
    await noesis_ingest.close_noesis_pool()
    await noesis_ingest.close_noesis_pool()
    assert closes == [1]


async def test_disabled_never_opens_pool(monkeypatch):
    async def boom(config):
        raise AssertionError("pool must not be created when disabled")

    monkeypatch.setattr(noesis_ingest, "_open_pool", boom)
    calls = patch_outcomes(monkeypatch, [])
    await ingest_noesis_batch(
        [content_item()], "bank-1", noesis_config(noesis_enabled=False),
        llm_config=llm_config(),
        extract_once_factory=FakeExtractOnceFactory(),
        pool_factory=None,  # production pool path, must never be reached
    )
    assert calls == []
