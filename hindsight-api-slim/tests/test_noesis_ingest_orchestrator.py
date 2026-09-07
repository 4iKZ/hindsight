"""Noesis ↔ orchestrator integration and Hermes glue tests.

Requirement 02 §17.1 / §17.9. Runs the real ``MemoryEngine.retain_batch_async``
against the embedded pg0 database with the mock LLM and a recording Noesis
seam, proving the production entry position, exactly-once ingestion under the
document_id grouping recursion, and failure isolation from the native retain.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

import hindsight_api.engine.retain.orchestrator as orchestrator
from tests.noesis_fakes import (
    CONTENT,
    FakeExtractOnceFactory,
    FakeStore,
    golden_fact_recursive,
    llm_config,
    noesis_config,
)


@pytest.fixture()
def noesis_recorder(monkeypatch):
    """Replace orchestrator.ingest_noesis_batch with a recording no-op."""
    import copy

    calls: list[dict] = []

    async def recorder(contents_dicts, bank_id, config, **kwargs):
        # retain_batch pops each item's "content" in place while streaming, so
        # snapshot the dicts at call time — a live reference would see them
        # emptied by the time the test asserts.
        calls.append(
            {
                "contents_dicts": copy.deepcopy(contents_dicts),
                "bank_id": bank_id,
                "config": config,
                **kwargs,
            }
        )

    monkeypatch.setattr(orchestrator, "ingest_noesis_batch", recorder)
    return calls


def _unique_bank() -> str:
    return f"noesis-orch-{uuid.uuid4().hex[:12]}"


async def test_retain_batch_invokes_noesis_once_for_all_items(memory, request_context, noesis_recorder):
    """Multi-document items are grouped; each group runs its own Memory
    Defense + Noesis pass exactly once per logical item."""
    bank_id = _unique_bank()
    contents = [
        {"content": "Alice works at Initech.", "document_id": "doc-a"},
        {"content": "Bob joined yesterday.", "document_id": "doc-b"},
        {"content": "Carol took a day off.", "document_id": "doc-c"},
    ]
    result = await memory.retain_batch_async(bank_id=bank_id, contents=contents, request_context=request_context)

    assert result is not None
    # each document_id group triggers exactly one ingestion pass over its own item
    assert len(noesis_recorder) == 3, "each grouped document must ingest its own item once"
    all_contents = [recorded["contents_dicts"][0]["content"] for recorded in noesis_recorder]
    assert sorted(all_contents) == sorted(item["content"] for item in contents)
    original_indices = sorted(recorded["contents_dicts"][0]["_noesis_item_index"] for recorded in noesis_recorder)
    assert original_indices == [0, 1, 2]
    # every recorded pass carries the resolved config with the noesis deployment switches
    for recorded in noesis_recorder:
        assert hasattr(recorded["config"], "noesis_enabled")
        assert hasattr(recorded["config"], "noesis_database_url")


async def test_noesis_failure_does_not_break_retain(memory, request_context, monkeypatch):
    async def failing_ingest(*args, **kwargs):
        raise RuntimeError("noesis exploded")

    monkeypatch.setattr(orchestrator, "ingest_noesis_batch", failing_ingest)
    bank_id = _unique_bank()
    result = await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": "Retain must survive a Noesis crash.", "document_id": "doc-x"}],
        request_context=request_context,
    )
    assert result is not None


async def test_retain_batch_ingests_with_resolved_config(memory, request_context, noesis_recorder):
    bank_id = _unique_bank()
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": "Config flows into ingestion."}],
        request_context=request_context,
    )
    assert len(noesis_recorder) == 1
    config = noesis_recorder[0]["config"]
    # the per-bank resolved config carries the noesis deployment switches
    assert hasattr(config, "noesis_enabled")
    assert hasattr(config, "noesis_database_url")
    assert hasattr(config, "noesis_schema")


async def test_retain_batch_passes_operation_and_document_ids(memory, request_context, noesis_recorder):
    bank_id = _unique_bank()
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": "Ids flow.", "document_id": "doc-1"}],
        request_context=request_context,
        document_id="batch-doc",
        operation_id="op-42",
    )
    recorded = noesis_recorder[0]
    assert recorded["document_id"] == "batch-doc"
    assert recorded["operation_id"] == "op-42"


# ---------------------------------------------------------------------------
# §17.9 / R02-07 Hermes → Hindsight → Noesis real-boundary contract
# ---------------------------------------------------------------------------

def _hermes_kwargs_payload(occurred_at: str) -> dict:
    """Invoke Hermes's real payload builder in its own project environment."""
    hermes_root = Path(__file__).resolve().parents[3] / "hermes-agent"
    script = f"""
import json
from plugins.memory.hindsight import HindsightMemoryProvider
p = HindsightMemoryProvider.__new__(HindsightMemoryProvider)
p._bank_id = 'hermes-bank'
p._retain_tags = []
p._observation_scopes = []
print(json.dumps(p._build_retain_kwargs({CONTENT!r}, metadata={{'source':'contract'}}, occurred_at={occurred_at!r})))
"""
    completed = subprocess.run(
        ["uv", "run", "python", "-c", script], cwd=hermes_root, check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout.strip())


def _real_hindsight_content_dict(hermes_payload: dict) -> dict:
    """Run the production HTTP boundary mapper, not a test-side replica."""
    from hindsight_api.api.http import MemoryItem, _memory_item_to_content_dict

    item = MemoryItem.model_validate(hermes_payload)
    return _memory_item_to_content_dict(item)


async def test_hermes_timestamp_flows_to_noesis_observed_at(monkeypatch):
    """Hermes payload → real Hindsight request model → Noesis observed_at.

    No LLM: HE returns the golden recursive fact. The timestamp→event_date
    conversion runs through the real ``MemoryItem`` validator, not a manual
    rename, so a production field-mapping break would fail this test.
    """
    import hindsight_api.engine.retain.noesis_ingest as noesis_ingest

    content_dict = _real_hindsight_content_dict(_hermes_kwargs_payload("2026-09-04T18:30:00+08:00"))

    store = FakeStore()
    seen: list = []

    def fake_extract(text, *, extract_once):
        seen.append(text)
        from hyperextract.noesis import ExtractionOutcome

        return ExtractionOutcome(components=[golden_fact_recursive()], alerts=[], attempts=1)

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", fake_extract)
    await noesis_ingest.ingest_noesis_batch(
        [content_dict],
        "hermes-bank",
        noesis_config(),
        llm_config=llm_config(),
        extract_once_factory=FakeExtractOnceFactory(),
        pool_factory=_fake_pool_factory(store),
    )

    assert seen == [CONTENT]
    data = next(iter(store.events.values()))["data"]
    assert data["observed_at"] == "2026-09-04T10:30:00Z"
    # no temporal modifier in the golden fact → event_time == observed_at
    event_time = next(iter(store.events.values()))["event_time"]
    assert event_time == datetime(2026, 9, 4, 10, 30, tzinfo=UTC)


async def test_hermes_empty_and_hypothesis_semantics(monkeypatch):
    """R02-07: the real boundary maps `[]` to no-event-no-alert and hypothesis
    to the deferred path only — verified through the real request model."""
    from hyperextract.noesis import ExtractionOutcome

    import hindsight_api.engine.retain.noesis_ingest as noesis_ingest

    store = FakeStore()

    def empty_extract(text, *, extract_once):
        return ExtractionOutcome(components=[], alerts=[], attempts=1)

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", empty_extract)
    await noesis_ingest.ingest_noesis_batch(
        [_real_hindsight_content_dict(_hermes_kwargs_payload("2026-09-04T18:30:00+08:00"))],
        "hermes-bank",
        noesis_config(),
        llm_config=llm_config(),
        extract_once_factory=FakeExtractOnceFactory(),
        pool_factory=_fake_pool_factory(store),
    )
    assert store.events == {}
    assert store.alerts == []


async def test_hermes_hypothesis_only_deferred(monkeypatch):
    """R02-07: hypothesis routed via the real boundary only writes
    hypothesis_deferred, never a fact row."""
    from hyperextract.noesis import ExtractionOutcome

    import hindsight_api.engine.retain.noesis_ingest as noesis_ingest
    from tests.noesis_fakes import golden_hypothesis

    store = FakeStore()

    def hyp_extract(text, *, extract_once):
        return ExtractionOutcome(components=[golden_hypothesis()], alerts=[], attempts=1)

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", hyp_extract)
    await noesis_ingest.ingest_noesis_batch(
        [_real_hindsight_content_dict(_hermes_kwargs_payload("2026-09-04T18:30:00+08:00"))],
        "hermes-bank",
        noesis_config(),
        llm_config=llm_config(),
        extract_once_factory=FakeExtractOnceFactory(),
        pool_factory=_fake_pool_factory(store),
    )
    assert store.events == {}
    assert store.atoms == {}
    deferred = store.alerts_by_code("hypothesis_deferred")
    assert len(deferred) == 1
    assert deferred[0]["stage"] == "hypothesis_routing"


def _fake_pool_factory(store):
    from tests.noesis_fakes import FakePool

    async def factory(_config):
        return FakePool(store)

    return factory


# ---------------------------------------------------------------------------
# R02-D1: Noesis ingestion runs AFTER Memory Defense screening/redaction
# ---------------------------------------------------------------------------

_REDACT_POLICY = {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]}}


async def _set_defense_policy(api_client, bank: str, updates: dict) -> None:
    r = await api_client.patch(f"/v1/default/banks/{bank}/config", json={"updates": updates})
    assert r.status_code == 200, r.text


async def test_noesis_ingests_redacted_text_not_verbatim_secret(api_client, monkeypatch):
    """When Memory Defense redacts a secret, Noesis receives the approved
    (redacted) content — never an unapproved verbatim copy (R02-D1)."""
    calls: list[list[dict]] = []

    async def recorder(contents_dicts, bank_id, config, **kwargs):
        calls.append(list(contents_dicts))

    monkeypatch.setattr(orchestrator, "ingest_noesis_batch", recorder)
    bank = f"md-noesis-{uuid.uuid4().hex[:8]}"
    await api_client.put(f"/v1/default/banks/{bank}", json={})
    await _set_defense_policy(api_client, bank, _REDACT_POLICY)

    secret = "ghp_" + "A" * 36
    r = await api_client.post(
        f"/v1/default/banks/{bank}/memories",
        json={"items": [{"content": f"my token is {secret}", "document_id": "doc-1"}]},
    )
    assert r.status_code == 200, r.text
    assert calls, "Noesis ingestion was never invoked"
    noesis_contents = [item.get("content", "") for item in calls[0]]
    # Noesis must never see the verbatim secret; its input must differ from the
    # raw submitted body (i.e. it consumed the Memory-Defense-approved text).
    assert all(secret not in content for content in noesis_contents), "Noesis saw the verbatim secret"
    assert all(content != f"my token is {secret}" for content in noesis_contents), (
        f"Noesis saw unredacted content; calls={calls}"
    )
    # Content was scrubbed into an approved shape (marker or wholesale redaction)
    assert all(content != "" for content in noesis_contents), f"unexpected empty ingestion: {calls}"


async def test_noesis_not_invoked_for_blocked_item(api_client, monkeypatch):
    """A Memory-Defense-blocked item is dropped before Noesis runs, so it never
    produces an unapproved copy (R02-D1)."""
    calls: list[list[dict]] = []

    async def recorder(contents_dicts, bank_id, config, **kwargs):
        calls.append(list(contents_dicts))

    monkeypatch.setattr(orchestrator, "ingest_noesis_batch", recorder)
    bank = f"md-block-{uuid.uuid4().hex[:8]}"
    await api_client.put(f"/v1/default/banks/{bank}", json={})
    block_policy = {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "block"}]}}
    await _set_defense_policy(api_client, bank, block_policy)

    secret = "sk-ant-" + "A" * 40
    r = await api_client.post(
        f"/v1/default/banks/{bank}/memories",
        json={"items": [{"content": f"leak: {secret}", "document_id": "doc-1"}]},
    )
    # blocked item → HTTP 422 per the memory-defense contract
    assert r.status_code == 422, r.text
    assert not calls, "Noesis ingestion must not run for a fully blocked item"
