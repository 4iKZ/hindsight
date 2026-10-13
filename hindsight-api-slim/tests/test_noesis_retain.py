"""Independent Noesis retain contract, using local fakes only."""

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from hindsight_api.api.http import NoesisRetainRequest, create_app
from hindsight_api.config import HindsightConfig
from hindsight_api.engine.memory_engine import MemoryEngine, NoesisDisabledError, NoesisRetainItemResult
from hindsight_api.engine.retain import noesis_ingest, orchestrator
from hindsight_api.engine.retain.noesis_ingest import NoesisItemResult
from hindsight_api.engine.retain.types import RetainContentDict
from hindsight_api.extensions import AuthenticationError, OperationValidationError, ValidationResult
from hindsight_api.extensions.memory_defense import DefenseAction, DefenseDecision
from hindsight_api.models import RequestContext
from tests.noesis_fakes import (
    CONTENT,
    FakeAnalyzer,
    FakeEmbeddingClient,
    FakeExtractOnceFactory,
    FakePool,
    FakeStore,
    embedding_factory_for,
    golden_fact_time,
    golden_hypothesis,
    he_alert,
    llm_config,
    noesis_config,
    outcome,
    pool_factory_for,
)


async def run_ingest(monkeypatch, extraction, *, summary=CONTENT, client=None, store=None):
    store = store or FakeStore()

    def extract(*args, **kwargs):
        if isinstance(extraction, Exception):
            raise extraction
        return extraction

    def summarize(text):
        if isinstance(summary, Exception):
            raise summary
        return summary

    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", extract)
    results = await noesis_ingest.ingest_noesis_batch(
        [{"content": CONTENT, "_noesis_item_index": 7}],
        "bank",
        noesis_config(),
        llm_config=llm_config(),
        analyzer=FakeAnalyzer(),
        pool_factory=pool_factory_for(store),
        extract_once_factory=FakeExtractOnceFactory(summarize_once=summarize),
        embedding_client_factory=embedding_factory_for(client or FakeEmbeddingClient()),
    )
    return SimpleNamespace(item=results[0], store=store)


@pytest.mark.parametrize(
    "components,status",
    [([], "no_events"), ([golden_hypothesis()], "hypothesis_only"), ([golden_fact_time()], "ingested")],
)
async def test_ingest_results(monkeypatch, components, status):
    result = await run_ingest(monkeypatch, outcome(components))
    assert result.item.index == 7
    assert result.item.status == status
    assert result.item.event_ids == list(result.store.events)
    assert result.item.alert_codes == [alert["alert_code"] for alert in result.store.alerts]


@pytest.mark.parametrize(
    "summary,status,code", [("", "empty_summary", None), (RuntimeError(), "failed", "summary_failed")]
)
async def test_summary_results(monkeypatch, summary, status, code):
    result = await run_ingest(monkeypatch, outcome(), summary=summary)
    assert result.item.status == status
    assert result.item.event_ids == []
    assert result.item.alert_codes == ([code] if code else [])


@pytest.mark.parametrize("extraction", [RuntimeError(), outcome(alerts=[he_alert("extraction_failed", "error")])])
async def test_extraction_failure_result(monkeypatch, extraction):
    result = await run_ingest(monkeypatch, extraction)
    assert result.item.status == "failed"
    assert result.item.alert_codes == ["extraction_failed"]


async def test_all_components_dropped(monkeypatch):
    result = await run_ingest(
        monkeypatch, outcome([golden_fact_time()]), client=FakeEmbeddingClient(health="unavailable")
    )
    assert result.item.status == "failed"
    assert result.item.event_ids == []
    assert result.item.alert_codes == ["anchor_context_service_unavailable"]


async def test_partial_success_keeps_ids_and_alerts(monkeypatch):
    store = FakeStore()
    store.profile = {"model_name": "wrong", "model_revision": "wrong", "dimension": 1024, "status": "ready"}
    original = noesis_ingest._route_component

    async def route(**kwargs):
        result = await original(**kwargs)
        store.profile = None
        return result

    monkeypatch.setattr(noesis_ingest, "_route_component", route)
    result = await run_ingest(monkeypatch, outcome([golden_fact_time(), golden_fact_time()]), store=store)
    assert result.item.status == "ingested"
    assert result.item.event_ids == list(store.events)
    assert len(result.item.event_ids) == 1
    assert result.item.alert_codes == ["embedding_profile_unavailable"]


async def test_alert_write_failure_does_not_claim_written_code(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(store, "_insert_alert", Mock(side_effect=RuntimeError("alert unavailable")))
    result = await run_ingest(monkeypatch, RuntimeError(), store=store)
    assert result.item.status == "failed"
    assert result.item.alert_codes == []


async def test_identity_vector_alert_keeps_committed_event(monkeypatch):
    fact = golden_fact_time()
    failures = {(atom.text, atom.type): "timeout" for atom in fact.atoms if atom.type in {"E", "P"}}
    result = await run_ingest(monkeypatch, outcome([fact]), client=FakeEmbeddingClient(identity_failures=failures))
    assert result.item.status == "ingested"
    assert result.item.event_ids == list(result.store.events)
    assert result.item.alert_codes == [alert["alert_code"] for alert in result.store.alerts]
    assert result.item.alert_codes


async def test_batch_results_keep_mixed_original_indices(monkeypatch):
    outcomes = iter([outcome(), outcome([golden_hypothesis()]), outcome([golden_fact_time()])])
    monkeypatch.setattr(noesis_ingest, "extract_noesis_components", lambda *args, **kwargs: next(outcomes))
    store = FakeStore()
    results = await noesis_ingest.ingest_noesis_batch(
        [{"content": CONTENT, "_noesis_item_index": index} for index in (2, 5, 9)],
        "bank",
        noesis_config(),
        llm_config=llm_config(),
        analyzer=FakeAnalyzer(),
        pool_factory=pool_factory_for(store),
        extract_once_factory=FakeExtractOnceFactory(),
        embedding_client_factory=embedding_factory_for(FakeEmbeddingClient()),
    )
    assert [item.index for item in results] == [2, 5, 9]
    assert [item.status for item in results] == ["no_events", "hypothesis_only", "ingested"]


@pytest.mark.parametrize("failure", ["pool", "preflight", "llm"])
async def test_setup_failure_returns_failed_results(monkeypatch, failure):
    store = FakeStore()
    if failure == "pool":
        monkeypatch.setattr(noesis_ingest, "_acquire_pool", AsyncMock(side_effect=RuntimeError()))
    elif failure == "preflight":
        monkeypatch.setattr(noesis_ingest, "_ensure_schema_ready", AsyncMock(return_value=False))
    factory = Mock(side_effect=RuntimeError()) if failure == "llm" else FakeExtractOnceFactory()
    results = await noesis_ingest.ingest_noesis_batch(
        [{"content": CONTENT}, {"content": CONTENT}],
        "bank",
        noesis_config(),
        llm_config=llm_config(),
        pool_factory=pool_factory_for(store),
        extract_once_factory=factory,
    )
    assert [item.status for item in results] == ["failed", "failed"]
    assert results[0].alert_codes == (["noesis_llm_config_invalid"] if failure == "llm" else [])
    assert results[1].alert_codes == []


async def test_disabled_batch_is_noop(monkeypatch):
    normalize = AsyncMock(side_effect=AssertionError("disabled must not normalize"))
    monkeypatch.setattr(noesis_ingest, "normalize_items", normalize)
    assert (
        await noesis_ingest.ingest_noesis_batch([{"content": CONTENT}], "bank", noesis_config(noesis_enabled=False))
        == []
    )
    normalize.assert_not_called()


def defense_pool():
    pool = FakePool(FakeStore())
    pool._wraps_backend = True
    return pool


def make_engine(*, enabled=True, decisions=()):
    engine = object.__new__(MemoryEngine)
    engine._authenticate_tenant = AsyncMock()
    engine._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(
            return_value=replace(HindsightConfig.from_env(), noesis_enabled=enabled, memory_defense={"enabled": True})
        )
    )
    engine._operation_validator = None
    engine._retain_llm_config = SimpleNamespace(with_config=Mock(return_value=llm_config()))
    engine._get_backend = AsyncMock(return_value=defense_pool())
    engine._memory_defense = SimpleNamespace(screen=AsyncMock(side_effect=list(decisions))) if decisions else None
    engine._webhook_manager = None
    engine._audit_logger = None
    # These native entrypoints must never be reached by the independent path.
    for name in (
        "_ensure_bank_exists",
        "submit_async_retain",
        "_submit_post_insert_maintenance",
        "_run_retain_execution",
    ):
        setattr(engine, name, AsyncMock(side_effect=AssertionError(name)))
    return engine


async def test_engine_disabled_authenticates_then_stops():
    engine = make_engine(enabled=False)
    with pytest.raises(NoesisDisabledError):
        await engine.retain_noesis_batch("bank", [{"content": CONTENT}], request_context=RequestContext())
    engine._authenticate_tenant.assert_awaited_once()
    engine._get_backend.assert_not_awaited()


async def test_engine_defense_redaction_indices_and_missing_results(monkeypatch):
    engine = make_engine(
        decisions=[
            DefenseDecision(action=DefenseAction.BLOCK, message="blocked"),
            DefenseDecision(action=DefenseAction.REDACT, redacted_content="safe"),
            DefenseDecision(action=DefenseAction.ALLOW),
        ]
    )
    ingest = AsyncMock(return_value=[NoesisItemResult(1, "ingested", [42], ["vector_warning"])])
    monkeypatch.setattr(noesis_ingest, "ingest_noesis_batch", ingest)
    results = await engine.retain_noesis_batch(
        "bank",
        [{"content": CONTENT, "tags": ["item"], "document_id": "source"}] * 3,
        request_context=RequestContext(),
        document_tags=["batch", "item"],
    )
    assert [item.status for item in results] == ["blocked", "ingested", "failed"]
    assert [item.index for item in results] == [0, 1, 2]
    assert results[1].event_ids == [42]
    sent = ingest.call_args.args[0]
    assert sent[0]["content"] == "safe"
    assert sent[0]["document_id"] == "source"
    assert [item["_noesis_item_index"] for item in sent] == [1, 2]
    assert set(engine._memory_defense.screen.call_args_list[0].kwargs["tags"]) == {"item", "batch"}
    assert ingest.call_args.kwargs["operation_id"] is None


async def test_engine_all_blocked_does_not_ingest(monkeypatch):
    engine = make_engine(decisions=[DefenseDecision(action=DefenseAction.BLOCK, message="blocked")])
    ingest = AsyncMock()
    monkeypatch.setattr(noesis_ingest, "ingest_noesis_batch", ingest)
    with pytest.raises(orchestrator.MemoryDefenseAllBlockedError):
        await engine.retain_noesis_batch("bank", [{"content": CONTENT}], request_context=RequestContext())
    ingest.assert_not_awaited()


async def test_engine_uses_validator_rewritten_contents(monkeypatch):
    engine = make_engine()
    engine._operation_validator = SimpleNamespace(
        validate_retain=AsyncMock(return_value=ValidationResult.accept_with(contents=[{"content": "rewritten"}]))
    )
    ingest = AsyncMock(return_value=[NoesisItemResult(0, "no_events", [], [])])
    monkeypatch.setattr(noesis_ingest, "ingest_noesis_batch", ingest)
    await engine.retain_noesis_batch("bank", [{"content": CONTENT}], request_context=RequestContext())
    assert ingest.call_args.args[0][0]["content"] == "rewritten"
    engine._operation_validator.validate_retain.assert_awaited_once()


async def test_engine_rejects_validator_changing_item_count(monkeypatch):
    engine = make_engine()
    engine._operation_validator = SimpleNamespace(
        validate_retain=AsyncMock(
            return_value=ValidationResult.accept_with(contents=[{"content": "a"}, {"content": "b"}])
        )
    )
    ingest = AsyncMock()
    monkeypatch.setattr(noesis_ingest, "ingest_noesis_batch", ingest)
    with pytest.raises(RuntimeError, match="number of retain items"):
        await engine.retain_noesis_batch("bank", [{"content": CONTENT}], request_context=RequestContext())
    ingest.assert_not_awaited()


async def test_engine_isolates_unexpected_ingest_exception(monkeypatch):
    engine = make_engine()
    monkeypatch.setattr(noesis_ingest, "ingest_noesis_batch", AsyncMock(side_effect=RuntimeError("batch failure")))
    results = await engine.retain_noesis_batch("bank", [{"content": CONTENT}], request_context=RequestContext())
    assert results == [NoesisRetainItemResult(0, "failed", [], [])]


async def test_shared_defense_preserves_notifications_and_dict_redaction(monkeypatch):
    decisions = [DefenseDecision(action=action, redacted_content="safe", message="policy") for action in DefenseAction]
    extension = SimpleNamespace(screen=AsyncMock(side_effect=decisions))
    notifications = []

    async def webhook(*args, **kwargs):
        notifications.append(("webhook", kwargs["decision"].action))

    async def audit(*args, **kwargs):
        notifications.append(("audit", kwargs["decision"].action))

    monkeypatch.setattr(orchestrator, "_fire_memory_defense_webhook", webhook)
    monkeypatch.setattr(orchestrator, "_audit_memory_defense", audit)
    raw: list[RetainContentDict] = [{"content": CONTENT}] * 3
    raw = [dict(item) for item in raw]
    result = await orchestrator.screen_contents_with_memory_defense(
        orchestrator._build_contents(raw, None),
        raw,
        pool=defense_pool(),
        schema="tenant",
        bank_id="bank",
        config=SimpleNamespace(memory_defense={"enabled": True}),
        memory_defense_extension=extension,
    )
    assert [item.content for item in result.contents] == [CONTENT, "safe"]
    assert result.contents_dicts[1]["content"] == "safe"
    assert result.blocked_violations[0].index == 2
    assert notifications == [
        (kind, action) for action in (DefenseAction.REDACT, DefenseAction.BLOCK) for kind in ("webhook", "audit")
    ]


@pytest.mark.parametrize(
    "items", [[], [{"content": CONTENT}] * 101, [{"content": " \n"}], [{"content": CONTENT, "timestamp": "unset"}]]
)
def test_request_validation(items):
    with pytest.raises(ValidationError):
        NoesisRetainRequest(items=items)


def test_request_accepts_datetime_null_and_ignores_native_fields():
    body = NoesisRetainRequest.model_validate(
        {"items": [{"content": CONTENT, "timestamp": None, "context": "ignored"}]}
    )
    assert body.items[0].timestamp is None
    assert "context" not in body.items[0].model_dump()
    stamp = datetime.now(UTC)
    assert NoesisRetainRequest(items=[{"content": CONTENT, "timestamp": stamp}]).items[0].timestamp == stamp


@pytest.fixture
def http_engine():
    return SimpleNamespace(
        audit_logger=None, _audit_logger=None, _operation_validator=None, retain_noesis_batch=AsyncMock()
    )


async def http_post(engine, body):
    app = create_app(engine, initialize_memory=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/v1/default/banks/bank/noesis/retain", json=body)


async def test_http_six_statuses(http_engine):
    statuses = ["ingested", "no_events", "hypothesis_only", "empty_summary", "failed", "blocked"]
    http_engine.retain_noesis_batch.return_value = [
        NoesisRetainItemResult(i, status, [42] if i == 0 else [], []) for i, status in enumerate(statuses)
    ]
    response = await http_post(http_engine, {"items": [{"content": CONTENT}] * 6})
    assert response.status_code == 200
    assert response.json()["items_count"] == 6
    assert [item["status"] for item in response.json()["results"]] == statuses
    assert response.json()["results"][0]["event_ids"] == [42]


@pytest.mark.parametrize(
    "error,code",
    [
        (NoesisDisabledError(), 503),
        (ValueError("bad input"), 400),
        (RuntimeError("internal"), 500),
        (orchestrator.MemoryDefenseAllBlockedError([orchestrator.BlockedViolation(0, "test", "blocked")]), 422),
    ],
)
async def test_http_error_mapping(http_engine, error, code):
    http_engine.retain_noesis_batch.side_effect = error
    response = await http_post(http_engine, {"items": [{"content": CONTENT}]})
    assert response.status_code == code
    if code == 503:
        assert response.json() == {"detail": {"error": "noesis_disabled"}}
    if code == 422:
        assert response.json()["detail"]["violations"][0]["index"] == 0


@pytest.mark.parametrize(
    "error,code", [(OperationValidationError("quota", 402), 402), (AuthenticationError("denied"), 401)]
)
async def test_http_authorization_error_mapping(http_engine, error, code):
    http_engine.retain_noesis_batch.side_effect = error
    response = await http_post(http_engine, {"items": [{"content": CONTENT}]})
    assert response.status_code == code


async def test_http_audit_captures_request_and_response(http_engine):
    audit = SimpleNamespace(
        action_allowed=Mock(return_value=True), should_log=AsyncMock(return_value=True), log_fire_and_forget=Mock()
    )
    http_engine.audit_logger = audit
    http_engine.retain_noesis_batch.return_value = [NoesisRetainItemResult(0, "no_events", [], [])]
    response = await http_post(http_engine, {"items": [{"content": CONTENT}]})
    assert response.status_code == 200
    entry = audit.log_fire_and_forget.call_args.args[0]
    assert entry.action == "noesis_retain"
    assert entry.request["items"][0]["content"] == CONTENT
    assert entry.response["results"][0]["status"] == "no_events"


async def test_http_retain_precheck_rejects_before_ingestion(http_engine):
    http_engine._authenticate_tenant = AsyncMock()
    http_engine._operation_validator = SimpleNamespace(
        precheck=AsyncMock(return_value=ValidationResult(allowed=False, reason="quota", status_code=402))
    )
    response = await http_post(http_engine, {"items": [{"content": CONTENT}]})
    assert response.status_code == 402
    assert http_engine._operation_validator.precheck.call_args.args[0].operation.value == "retain"
    http_engine.retain_noesis_batch.assert_not_awaited()


@pytest.mark.parametrize(
    "items", [[], [{"content": CONTENT}] * 101, [{"content": " "}], [{"content": CONTENT, "timestamp": "bad"}]]
)
async def test_http_validation_does_not_call_engine(http_engine, items):
    response = await http_post(http_engine, {"items": items})
    assert response.status_code == 422
    http_engine.retain_noesis_batch.assert_not_awaited()


def test_native_hook_is_static_config():
    assert "noesis_native_hook_enabled" not in HindsightConfig.get_configurable_fields()
