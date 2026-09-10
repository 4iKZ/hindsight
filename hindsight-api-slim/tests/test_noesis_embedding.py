"""Noesis shared bge embedding client + write-path tests (requirements 03 + 05).

Offline only: the bge HTTP layer is exercised through ``httpx.MockTransport``
(§15.1), and the ingest flow through ``FakeEmbeddingClient`` + ``FakeStore``
(§15.2/§15.3). No test in this file touches the real network or a real
database. Requirement 05 §13.2 adds the ``embed_context`` client tests.
"""

from __future__ import annotations

import json

import httpx
import pytest

from hindsight_api.engine.retain.noesis_embedding import (
    EmbeddingConfigInvalid,
    EmbeddingError,
    EmbeddingServiceUnavailable,
    NoesisEmbeddingClient,
    normalize_model_basename,
)

DIM = 1024


def _vector(seed: float = 0.01, dim: int = DIM) -> list[float]:
    return [seed] * dim


def _ok_health(model: str = "BAAI/bge-m3", dim: int = DIM) -> dict:
    return {
        "status": "ok",
        "model": model,
        "dim": dim,
        "device": "cuda:1",
        "endpoints": ["/normalize/token", "/normalize/entity", "/normalize/predicate", "/normalize/sentence"],
    }


def _client(handler, *, model: str = "bge-m3", api_key: str = "", max_retries: int = 1) -> NoesisEmbeddingClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://bge.test")
    return NoesisEmbeddingClient(
        base_url="http://bge.test",
        model=model,
        revision="bge-m3-1024-v1",
        dimension=DIM,
        timeout_seconds=2.0,
        max_retries=max_retries,
        api_key=api_key,
        http_client=http,
    )


class _Recorder:
    """MockTransport handler factory: records requests, serves scripted replies."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if self.replies else httpx.Response(200, json=_ok_health())
        if isinstance(reply, Exception):
            raise reply
        return reply


def _embed_response(dim: int = DIM) -> httpx.Response:
    return httpx.Response(200, json={"embedding": _vector(), "dim": dim, "model": "bge-m3"})


# ---------------------------------------------------------------------------
# §15.1 endpoint mapping, request shape, response parsing (identity)
# ---------------------------------------------------------------------------

async def test_entity_endpoint_exact_request_shape():
    recorder = _Recorder([_embed_response()])
    client = _client(recorder)
    result = await client.embed_identity("苹果", "E")

    assert result == _vector()
    (request,) = recorder.requests
    assert request.method == "POST"
    assert request.url.path == "/normalize/entity"
    assert json.loads(request.content) == {"term": "苹果"}  # no threshold, no prefix


async def test_predicate_endpoint_exact_request_shape():
    recorder = _Recorder([_embed_response()])
    client = _client(recorder)
    await client.embed_identity("买", "P")

    (request,) = recorder.requests
    assert request.url.path == "/normalize/predicate"
    assert json.loads(request.content) == {"term": "买"}


async def test_response_embedding_field_and_dimension_pass():
    recorder = _Recorder([httpx.Response(200, json={"embedding": _vector(0.5), "dim": DIM, "model": "bge-m3"})])
    client = _client(recorder)
    assert await client.embed_identity("妈妈", "E") == _vector(0.5)


async def test_response_wrong_dimension_fails():
    recorder = _Recorder([httpx.Response(200, json={"embedding": [0.1] * 512, "dim": 512, "model": "bge-m3"})])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_identity("妈妈", "E")
    assert exc_info.value.error_kind == "bad_dimension"


async def test_response_non_finite_component_fails():
    payload = json.dumps({"embedding": [float("nan")] + [0.0] * (DIM - 1)}).encode()
    recorder = _Recorder(
        [httpx.Response(200, content=payload, headers={"content-type": "application/json"})]
    )
    client = _client(recorder)

    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_identity("苹果", "E")

    assert exc_info.value.error_kind == "bad_payload"


async def test_response_missing_embedding_field_fails():
    recorder = _Recorder([httpx.Response(200, json={"dim": DIM})])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_identity("妈妈", "E")
    assert exc_info.value.error_kind == "bad_payload"


async def test_response_invalid_json_fails():
    recorder = _Recorder([httpx.Response(200, content=b"not json")])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_identity("妈妈", "E")
    assert exc_info.value.error_kind == "bad_payload"


# ---------------------------------------------------------------------------
# §15.1 timeout / retry policy (identity)
# ---------------------------------------------------------------------------

async def test_transport_error_retried_once_then_succeeds():
    recorder = _Recorder([httpx.ConnectTimeout("slow"), _embed_response()])
    client = _client(recorder)
    assert await client.embed_identity("妈妈", "E") == _vector()
    assert len(recorder.requests) == 2


async def test_transport_error_exhausts_retry_budget():
    recorder = _Recorder([httpx.ConnectTimeout("slow"), httpx.ConnectTimeout("slow")])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_identity("妈妈", "E")
    assert exc_info.value.error_kind == "timeout"
    assert len(recorder.requests) == 1 + 1  # one retry, no more


async def test_5xx_retried_then_fails_as_http_5xx():
    recorder = _Recorder([httpx.Response(503), httpx.Response(500)])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_identity("妈妈", "E")
    assert exc_info.value.error_kind == "http_5xx"
    assert len(recorder.requests) == 2


async def test_5xx_retried_then_succeeds():
    recorder = _Recorder([httpx.Response(502), _embed_response()])
    client = _client(recorder)
    assert await client.embed_identity("妈妈", "E") == _vector()


async def test_4xx_never_retried():
    recorder = _Recorder([httpx.Response(400)])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_identity("妈妈", "E")
    assert exc_info.value.error_kind == "http_4xx"
    assert len(recorder.requests) == 1


async def test_422_is_contract_violation_never_retried():
    recorder = _Recorder([httpx.Response(422)])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_identity("妈妈", "E")
    assert exc_info.value.error_kind == "contract_violation"
    assert len(recorder.requests) == 1


async def test_zero_retries_means_single_attempt():
    recorder = _Recorder([httpx.ConnectTimeout("slow")])
    client = _client(recorder, max_retries=0)
    with pytest.raises(EmbeddingError):
        await client.embed_identity("妈妈", "E")
    assert len(recorder.requests) == 1


# ---------------------------------------------------------------------------
# §15.1 /health gate and its cache semantics
# ---------------------------------------------------------------------------

async def test_health_ok_cached_after_first_call():
    recorder = _Recorder([httpx.Response(200, json=_ok_health())])
    client = _client(recorder)
    await client.ensure_ready()
    await client.ensure_ready()
    assert len(recorder.requests) == 1  # once per process, not per fact


async def test_health_dim_mismatch_is_config_invalid_and_cached():
    recorder = _Recorder([httpx.Response(200, json=_ok_health(dim=512))])
    client = _client(recorder)
    with pytest.raises(EmbeddingConfigInvalid):
        await client.ensure_ready()
    with pytest.raises(EmbeddingConfigInvalid):
        await client.ensure_ready()
    assert len(recorder.requests) == 1  # config failures are cached (process-level off)


async def test_health_status_not_ok_is_config_invalid():
    recorder = _Recorder([httpx.Response(200, json=_ok_health() | {"status": "loading"})])
    client = _client(recorder)
    with pytest.raises(EmbeddingConfigInvalid):
        await client.ensure_ready()


async def test_health_model_mismatch_rejected_by_exact_basename():
    """'bge-m3-evil' must NOT pass a contains/substring check for 'bge-m3'."""
    recorder = _Recorder([httpx.Response(200, json=_ok_health(model="BAAI/bge-m3-evil"))])
    client = _client(recorder)
    with pytest.raises(EmbeddingConfigInvalid):
        await client.ensure_ready()


async def test_health_model_basename_equivalence():
    """Configured 'bge-m3' and service 'BAAI/bge-m3' are the same generation."""
    recorder = _Recorder([httpx.Response(200, json=_ok_health(model="BAAI/bge-m3"))])
    client = _client(recorder, model="bge-m3")
    await client.ensure_ready()  # no exception

    recorder2 = _Recorder([httpx.Response(200, json=_ok_health(model="bge-m3"))])
    client2 = _client(recorder2, model="BAAI/bge-m3")
    await client2.ensure_ready()


async def test_health_transport_failure_not_cached_and_recovers():
    recorder = _Recorder([httpx.ConnectError("down"), httpx.Response(200, json=_ok_health())])
    client = _client(recorder)
    with pytest.raises(EmbeddingServiceUnavailable):
        await client.ensure_ready()
    await client.ensure_ready()  # transport failures are not cached: retried next time
    assert len(recorder.requests) == 2


async def test_health_non_200_treated_as_unavailable_not_config_invalid():
    recorder = _Recorder([httpx.Response(503), httpx.Response(200, json=_ok_health())])
    client = _client(recorder)
    with pytest.raises(EmbeddingServiceUnavailable):
        await client.ensure_ready()
    await client.ensure_ready()


# ---------------------------------------------------------------------------
# §15.1 auth header, blank rejection, type guard (identity)
# ---------------------------------------------------------------------------

async def test_bearer_header_sent_when_api_key_configured():
    recorder = _Recorder([_embed_response()])
    client = _client(recorder, api_key="secret-token")
    await client.embed_identity("苹果", "E")
    assert recorder.requests[0].headers["Authorization"] == "Bearer secret-token"


async def test_no_bearer_header_without_api_key():
    recorder = _Recorder([_embed_response()])
    client = _client(recorder)
    await client.embed_identity("苹果", "E")
    assert "Authorization" not in recorder.requests[0].headers


async def test_blank_literal_rejected_before_any_http():
    recorder = _Recorder([])
    client = _client(recorder)
    for blank in ("", "   ", " \n\t "):
        with pytest.raises(EmbeddingError) as exc_info:
            await client.embed_identity(blank, "E")
        assert exc_info.value.error_kind == "empty_literal"
    assert recorder.requests == []  # zero HTTP


async def test_g_type_literal_rejected_before_any_http():
    recorder = _Recorder([])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_identity("周末计划", "G")
    assert exc_info.value.error_kind == "unsupported_atom_type"
    assert recorder.requests == []


# ---------------------------------------------------------------------------
# model basename normalization helper
# ---------------------------------------------------------------------------

def test_normalize_model_basename():
    assert normalize_model_basename("BAAI/bge-m3") == "bge-m3"
    assert normalize_model_basename("bge-m3") == "bge-m3"
    assert normalize_model_basename("Org/Sub/BGE-M3") == "bge-m3"
    assert normalize_model_basename("bge-m3-evil") != normalize_model_basename("bge-m3")


# ---------------------------------------------------------------------------
# §13.2 Context HTTP client (requirement 05)
# ---------------------------------------------------------------------------

async def test_context_endpoint_and_exact_request_shape():
    """§13.2 #1/#2: POST /normalize/sentence with body exactly {"text": ...}."""
    recorder = _Recorder([_embed_response()])
    client = _client(recorder)
    result = await client.embed_context("妈妈 让 小明")

    assert result == _vector()
    (request,) = recorder.requests
    assert request.method == "POST"
    assert request.url.path == "/normalize/sentence"
    assert json.loads(request.content) == {"text": "妈妈 让 小明"}  # exactly one key, no extras


async def test_context_response_embedding_field_and_dimension_pass():
    """§13.2 #3: the 1024-dim embedding field parses; canonical_text is ignored."""
    recorder = _Recorder(
        [
            httpx.Response(
                200,
                json={
                    "embedding": _vector(0.25),
                    "dim": DIM,
                    "model": "bge-m3",
                    "canonical_text": "妈妈让小明",
                },
            )
        ]
    )
    client = _client(recorder)
    assert await client.embed_context("妈妈 让 小明") == _vector(0.25)


async def test_context_timeout_retried_then_succeeds():
    """§13.2 #4: timeout is retried per the configured budget."""
    recorder = _Recorder([httpx.ConnectTimeout("slow"), _embed_response()])
    client = _client(recorder)
    assert await client.embed_context("妈妈 让 小明") == _vector()
    assert len(recorder.requests) == 2


async def test_context_transport_error_exhausts_retry_budget():
    """§13.2 #4: transport errors exhaust the budget and surface the kind."""
    recorder = _Recorder([httpx.ConnectError("down"), httpx.ConnectError("down")])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_context("妈妈 让 小明")
    assert exc_info.value.error_kind == "transport"
    assert len(recorder.requests) == 1 + 1


async def test_context_5xx_retried_then_fails_as_http_5xx():
    """§13.2 #4: 5xx is retried, then reported."""
    recorder = _Recorder([httpx.Response(503), httpx.Response(500)])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_context("妈妈 让 小明")
    assert exc_info.value.error_kind == "http_5xx"
    assert len(recorder.requests) == 2


async def test_context_4xx_never_retried():
    """§13.2 #5: 4xx fails immediately, zero retries."""
    recorder = _Recorder([httpx.Response(400)])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_context("妈妈 让 小明")
    assert exc_info.value.error_kind == "http_4xx"
    assert len(recorder.requests) == 1


async def test_context_422_never_retried():
    """§13.2 #5: 422 is a contract violation, never retried."""
    recorder = _Recorder([httpx.Response(422)])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_context("妈妈 让 小明")
    assert exc_info.value.error_kind == "contract_violation"
    assert len(recorder.requests) == 1


async def test_context_non_json_fails():
    """§13.2 #6: a non-JSON body is a bad_payload failure."""
    recorder = _Recorder([httpx.Response(200, content=b"not json")])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_context("妈妈 让 小明")
    assert exc_info.value.error_kind == "bad_payload"


async def test_context_missing_embedding_fails():
    """§13.2 #7: a payload without the embedding field fails."""
    recorder = _Recorder([httpx.Response(200, json={"dim": DIM, "canonical_text": "妈妈让小明"})])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_context("妈妈 让 小明")
    assert exc_info.value.error_kind == "bad_payload"


async def test_context_wrong_dimension_fails():
    """§13.2 #8: a wrong-dimension embedding fails."""
    recorder = _Recorder([httpx.Response(200, json={"embedding": [0.1] * 512, "dim": 512})])
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_context("妈妈 让 小明")
    assert exc_info.value.error_kind == "bad_dimension"


async def test_context_non_finite_component_fails():
    """§13.2 #9: NaN/Infinity components fail; nothing is faked."""
    payload = json.dumps({"embedding": [float("inf")] + [0.0] * (DIM - 1)}).encode()
    recorder = _Recorder(
        [httpx.Response(200, content=payload, headers={"content-type": "application/json"})]
    )
    client = _client(recorder)
    with pytest.raises(EmbeddingError) as exc_info:
        await client.embed_context("妈妈 让 小明")
    assert exc_info.value.error_kind == "bad_payload"


async def test_context_never_calls_token_endpoint():
    """§13.2 #10: /normalize/token is never touched."""
    recorder = _Recorder([_embed_response()])
    client = _client(recorder)
    await client.embed_context("妈妈 让 小明")
    assert all(request.url.path != "/normalize/token" for request in recorder.requests)


async def test_context_never_calls_identity_endpoints():
    """§13.2 #11: context vectors never go through the identity endpoints."""
    recorder = _Recorder([_embed_response()])
    client = _client(recorder)
    await client.embed_context("妈妈 让 小明")
    paths = [request.url.path for request in recorder.requests]
    assert "/normalize/entity" not in paths
    assert "/normalize/predicate" not in paths


async def test_context_and_identity_share_health_and_client_lifecycle():
    """§13.2 #12: one /health probe and one client serve both embed kinds."""
    recorder = _Recorder(
        [httpx.Response(200, json=_ok_health()), _embed_response(), _embed_response()]
    )
    client = _client(recorder)
    await client.ensure_ready()
    await client.embed_identity("苹果", "E")
    await client.embed_context("妈妈 让 小明")

    health_probes = [request for request in recorder.requests if request.url.path == "/health"]
    assert len(health_probes) == 1  # cached state shared by both kinds
    paths = [request.url.path for request in recorder.requests]
    assert paths.count("/normalize/entity") == 1
    assert paths.count("/normalize/sentence") == 1


async def test_context_failures_never_leak_api_key_or_vectors(caplog):
    """§13.2 #13: neither exceptions nor logs carry the API key or vector data."""
    secret = "secret-token"
    payload = json.dumps({"embedding": [0.123456789] + [0.0] * (DIM - 1)}).encode()
    recorder = _Recorder([httpx.Response(400, content=payload)])
    client = _client(recorder, api_key=secret)

    with caplog.at_level("ERROR"):
        with pytest.raises(EmbeddingError) as exc_info:
            await client.embed_context("妈妈 让 小明")

    message = str(exc_info.value)
    assert secret not in message
    assert "Bearer" not in message
    assert "0.123456789" not in message
    for record in caplog.records:
        assert secret not in record.getMessage()
        assert "Bearer" not in record.getMessage()
        assert "0.123456789" not in record.getMessage()


# ---------------------------------------------------------------------------
# §15.2 / §15.3 ingest flow + degradation (fake bge + fake store)
# ---------------------------------------------------------------------------

from datetime import timedelta  # noqa: E402

import hindsight_api.engine.retain.noesis_ingest as noesis_ingest  # noqa: E402
from hindsight_api.engine.retain.noesis_ingest import ingest_noesis_batch  # noqa: E402
from tests.noesis_fakes import (  # noqa: E402
    CONTENT,
    OBSERVED_AT,
    FakeEmbeddingClient,
    FakeExtractOnceFactory,
    FakeStore,
    embedding_factory_for,
    fake_identity_vector,
    golden_fact_recursive,
    llm_config,
    noesis_config,
    outcome,
    pool_factory_for,
)

_GOLDEN_LITERALS = [("小明", "E"), ("没写", "P"), ("作业", "E"), ("揍", "P")]


def _one_fact_contents(event_date=OBSERVED_AT):
    return [{"content": CONTENT, "event_date": event_date, "document_id": "doc-1"}]


async def _run(store, contents, components, *, identity=None, monkeypatch=None, **kwargs):
    """Ingest with a fake bge client (healthy unless overridden)."""
    client = identity if identity is not None else FakeEmbeddingClient()
    config = kwargs.pop("config", noesis_config())

    def fake_extract(text, *, extract_once):
        return outcome(list(components or []))

    if monkeypatch is not None:
        monkeypatch.setattr(noesis_ingest, "extract_noesis_components", fake_extract)
    else:
        noesis_ingest.extract_noesis_components = fake_extract
    try:
        await ingest_noesis_batch(
            contents,
            "bank-1",
            config,
            llm_config=llm_config(),
            extract_once_factory=FakeExtractOnceFactory(),
            pool_factory=pool_factory_for(store),
            embedding_client_factory=embedding_factory_for(client),
            **kwargs,
        )
    finally:
        if monkeypatch is None:
            noesis_ingest.extract_noesis_components = _ORIGINAL_EXTRACT
    return client


_ORIGINAL_EXTRACT = noesis_ingest.extract_noesis_components


# -- §15.2 write path --------------------------------------------------------

async def test_new_ep_atoms_created_with_identity_vectors():
    store = FakeStore()
    client = await _run(store, _one_fact_contents(), [golden_fact_recursive()])

    assert sorted(client.identity_calls) == sorted(_GOLDEN_LITERALS)  # one call per distinct E/P literal
    for text, atom_type in _GOLDEN_LITERALS:
        expected = fake_identity_vector(text, atom_type)
        assert store.embedding_of(text, atom_type) == expected
        assert len(store.embedding_of(text, atom_type)) == 1024


async def test_existing_vector_never_recomputed_never_overwritten():
    store = FakeStore()
    await _run(store, _one_fact_contents(), [golden_fact_recursive()])
    original = {literal: store.embedding_of(*literal) for literal in _GOLDEN_LITERALS}

    # A DIFFERENT event (new observed_at) re-mentions the same literals.
    client2 = FakeEmbeddingClient()
    await _run(store, _one_fact_contents(OBSERVED_AT + timedelta(days=1)), [golden_fact_recursive()], identity=client2)

    assert client2.identity_calls == []  # precheck saw has_embedding → zero bge
    for literal, vector in original.items():
        assert store.embedding_of(*literal) == vector  # untouched
    assert len(store.atoms) == 4  # upserts stay one row per typed literal


async def test_null_embedding_self_heals_on_next_occurrence():
    store = FakeStore()
    failing = FakeEmbeddingClient(identity_failures={literal: "timeout" for literal in _GOLDEN_LITERALS})
    await _run(store, _one_fact_contents(), [golden_fact_recursive()], identity=failing)
    assert all(store.embedding_of(*literal) is None for literal in _GOLDEN_LITERALS)

    healing = FakeEmbeddingClient()
    await _run(store, _one_fact_contents(OBSERVED_AT + timedelta(days=1)), [golden_fact_recursive()], identity=healing)

    assert sorted(healing.identity_calls) == sorted(_GOLDEN_LITERALS)  # NULL → re-embedded
    for text, atom_type in _GOLDEN_LITERALS:
        assert store.embedding_of(text, atom_type) == fake_identity_vector(text, atom_type)
    assert len(store.atoms) == 4


async def test_same_event_same_literal_one_bge_one_upsert():
    store = FakeStore()
    client = await _run(store, _one_fact_contents(), [golden_fact_recursive()])

    assert client.identity_calls.count(("小明", "E")) == 1  # 小明 occurs twice in the golden fact
    upserts = [args for kind, sql, args in store.calls if kind == "fetchrow" and "INSERT" in sql and ".atoms" in sql]
    assert len([args for args in upserts if args[0] == "小明" and args[1] == "E"]) == 1


async def test_g_atom_never_embedded_component_dropped():
    """Requirement 05 §2.3/§5.1: a G atom is not routable, so the component is
    dropped with an anchor_frame_invalid alert — the G literal is never sent
    to bge and never registered."""
    import json as _json

    from hyperextract.noesis import FactComponent

    from tests.noesis_fakes import GOLDEN_FACT_RECURSIVE_JSON

    payload = _json.loads(_json.dumps(GOLDEN_FACT_RECURSIVE_JSON))
    payload["atoms"].append(
        {"pos": 6, "text": "周末计划", "type": "G", "role": "modifier", "target_occ": 4, "resolved": None}
    )
    store = FakeStore()
    client = await _run(store, _one_fact_contents(), [FactComponent.model_validate(payload)])

    assert ("周末计划", "G") not in client.identity_calls  # G never gets a vector
    assert store.embedding_of("周末计划", "G") is None
    assert store.events == {}  # the whole component is dropped (05 §5.1)
    assert len(store.alerts_by_code("anchor_frame_invalid")) == 1


async def test_duplicate_input_new_event_zero_reembed():
    """04A: the same input again lands a NEW event (sequence number), but the
    already-embedded atoms are neither re-embedded nor overwritten."""
    store = FakeStore()
    await _run(store, _one_fact_contents(), [golden_fact_recursive()])
    original = {literal: store.embedding_of(*literal) for literal in _GOLDEN_LITERALS}

    duplicate_client = FakeEmbeddingClient()
    await _run(store, _one_fact_contents(), [golden_fact_recursive()], identity=duplicate_client)

    assert len(store.events) == 2  # duplicate input is a new event (04A contract)
    assert duplicate_client.identity_calls == []  # precheck saw has_embedding → zero bge
    for literal, vector in original.items():
        assert store.embedding_of(*literal) == vector  # untouched
    assert len(store.atoms) == 4  # upserts stay one row per typed literal


async def test_component_without_ep_literals_no_precheck_no_bge():
    from hyperextract.noesis import FactComponent

    payload = {
        "utterance_type": "fact",
        "atoms": [
            {"pos": 1, "text": "概念甲", "type": "G", "role": "agent", "target_occ": 2, "resolved": None},
            {"pos": 2, "text": "概念乙", "type": "G", "role": "predicate", "target_occ": None, "resolved": None},
        ],
        "tree": {
            "predicate": "概念乙",
            "agent": [{"text": "概念甲", "modifier": [], "implied": False}],
            "patient": [],
            "modifier": [],
            "nested": [],
            "conditional": [],
        },
    }
    store = FakeStore()
    client = await _run(store, _one_fact_contents(), [FactComponent.model_validate(payload)])

    assert client.identity_calls == []
    assert store.fetch_calls("unnest") == []  # zero precheck
    assert store.embedding_of("概念甲", "G") is None
    # Requirement 05 §2.3/§5.1: G atoms are not routable, so the whole
    # component is dropped with a frame alert instead of persisting.
    assert store.events == {}
    assert len(store.alerts_by_code("anchor_frame_invalid")) == 1


# -- §15.3 degradation -------------------------------------------------------

async def test_bge_total_transport_failure_null_registration_and_unavailable_alert():
    store = FakeStore()
    failing = FakeEmbeddingClient(identity_failures={literal: "timeout" for literal in _GOLDEN_LITERALS})
    await _run(store, _one_fact_contents(), [golden_fact_recursive()], identity=failing)

    assert len(store.events) == 1  # the fact still registers
    assert all(store.embedding_of(*literal) is None for literal in _GOLDEN_LITERALS)
    alerts = store.alerts_by_code("identity_vector_unavailable")
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["stage"] == "identity_vector"
    assert alert["severity"] == "warning"
    assert alert["event_id"] is not None  # bound to the committed event
    details = alert["details"]
    assert details["attempted"] == 4 and details["succeeded"] == 0
    assert {tuple(sorted((f["text"], f["atom_type"]))) for f in details["failed_atoms"]} == {
        tuple(sorted(literal)) for literal in _GOLDEN_LITERALS
    }
    assert details["model"] == "bge-m3" and details["base_url"] == "http://identity.test"


async def test_bge_partial_failure_exact_failed_subset_in_alert():
    store = FakeStore()
    failing = FakeEmbeddingClient(identity_failures={("揍", "P"): "http_5xx"})
    await _run(store, _one_fact_contents(), [golden_fact_recursive()], identity=failing)

    assert store.embedding_of("揍", "P") is None
    for literal in ("小明", "E"), ("没写", "P"), ("作业", "E"):
        assert store.embedding_of(*literal) == fake_identity_vector(*literal)
    alerts = store.alerts_by_code("identity_vector_failed")
    assert len(alerts) == 1
    details = alerts[0]["details"]
    assert details["failed_atoms"] == [{"text": "揍", "atom_type": "P", "error_kind": "http_5xx"}]
    assert details["attempted"] == 4 and details["succeeded"] == 3


async def test_health_config_invalid_aborts_component_with_alert():
    """Requirement 05 §8.1: Context is a hard precondition — a /health config
    mismatch now drops the whole component (anchor_context_config_invalid)
    instead of degrading to NULL identity vectors."""
    store = FakeStore()
    invalid = FakeEmbeddingClient(health="config_invalid")
    await _run(store, _one_fact_contents(), [golden_fact_recursive()], identity=invalid)

    assert store.events == {}  # zero fact state, unlike the identity degradation
    assert store.atoms == {}
    assert store.event_atoms == []
    assert store.anchors_rows() == []
    assert invalid.identity_calls == []
    alerts = store.alerts_by_code("anchor_context_config_invalid")
    assert len(alerts) == 1
    assert alerts[0]["stage"] == "anchor_context"
    assert alerts[0]["severity"] == "error"
    assert alerts[0]["details"]["exception_type"] == "EmbeddingConfigInvalid"


async def test_profile_claimed_on_empty_store():
    store = FakeStore()
    await _run(store, _one_fact_contents(), [golden_fact_recursive()])

    assert store.profile == {
        "embedding_kind": "identity",
        "model_name": "bge-m3",
        "model_revision": "bge-m3-1024-v1",
        "dimension": 1024,
        "status": "ready",
    }


async def test_profile_mismatch_drops_component_with_fixed_alert():
    """05A §5.3: a generation mismatch now refuses the whole component (zero
    facts, zero bge HTTP) with the fixed embedding_profile_unavailable alert
    — the requirement-03 NULL-vector degradation no longer applies."""
    store = FakeStore()
    store.profile = {
        "embedding_kind": "identity",
        "model_name": "other-model",
        "model_revision": "other-r0",
        "dimension": 1024,
        "status": "ready",
    }
    client = await _run(store, _one_fact_contents(), [golden_fact_recursive()])

    assert store.events == {}  # 05A: whole component dropped
    assert store.atoms == {}
    assert store.event_atoms == []
    assert store.anchors == {}
    assert client.identity_calls == [] and client.context_calls == []  # zero bge
    alerts = store.alerts_by_code("embedding_profile_unavailable")
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "error"
    assert alerts[0]["details"]["reason"] == "generation_mismatch"
    assert alerts[0]["details"]["db_profile"]["model_name"] == "other-model"
    assert alerts[0]["details"]["config_profile"]["model_name"] == "bge-m3"


async def test_profile_rebuilding_drops_component_with_fixed_alert():
    store = FakeStore()
    store.profile = {
        "embedding_kind": "identity",
        "model_name": "bge-m3",
        "model_revision": "bge-m3-1024-v1",
        "dimension": 1024,
        "status": "rebuilding",
    }
    client = await _run(store, _one_fact_contents(), [golden_fact_recursive()])

    assert store.events == {}
    assert client.identity_calls == [] and client.context_calls == []
    alerts = store.alerts_by_code("embedding_profile_unavailable")
    assert len(alerts) == 1
    assert alerts[0]["details"]["reason"] == "rebuilding"


async def test_profile_flipped_to_rebuilding_after_gate_rolls_back_fact():
    """A rebuild starting after the outer gate is caught by the in-transaction
    FOR SHARE fence: the whole component rolls back with zero facts (05A
    §5.5) instead of the requirement-03 vectors-discarded degradation."""
    store = FakeStore()
    store.profile = {
        "embedding_kind": "identity",
        "model_name": "bge-m3",
        "model_revision": "bge-m3-1024-v1",
        "dimension": 1024,
        "status": "ready",
    }
    client = FakeEmbeddingClient()
    switched = False

    async def start_rebuild_before_first_transaction_statement():
        nonlocal switched
        if not switched:
            switched = True
            store.profile["status"] = "rebuilding"

    store.interleave = start_rebuild_before_first_transaction_statement
    await _run(store, _one_fact_contents(), [golden_fact_recursive()], identity=client)

    assert client.identity_calls  # gate allowed, identity bge ran
    assert store.events == {}  # fence rolled the whole component back
    assert store.atoms == {}
    assert store.event_atoms == []
    assert store.anchors == {}
    alerts = store.alerts_by_code("embedding_profile_unavailable")
    assert len(alerts) == 1
    assert alerts[0]["details"]["reason"] == "rebuilding"
    guard_calls = [
        call
        for call in store.calls
        if ".embedding_profiles" in call[1] and "FOR SHARE" in call[1]
    ]
    assert len(guard_calls) == 1


async def test_profile_missing_but_vectors_present_refuses_to_guess():
    """05A §5.2: missing profile + existing E/P vectors → refuse the component
    (missing_with_existing_data); the store is never claimed over them."""
    store = FakeStore()
    store.atoms[("旧词", "E")] = {"atom_id": 42, "embedding": fake_identity_vector("旧词", "E")}
    store.profile = None
    client = await _run(store, _one_fact_contents(), [golden_fact_recursive()])

    assert store.events == {}  # 05A: whole component dropped
    assert client.identity_calls == [] and client.context_calls == []
    assert store.profile is None  # never guesses the source of existing vectors
    alerts = store.alerts_by_code("embedding_profile_unavailable")
    assert len(alerts) == 1
    assert alerts[0]["details"]["reason"] == "missing_with_existing_data"
    assert "db_profile" not in alerts[0]["details"]


async def test_vector_alert_write_failure_only_logs():
    store = FakeStore()
    failing = FakeEmbeddingClient(identity_failures={literal: "timeout" for literal in _GOLDEN_LITERALS})
    store.fail_after = {".ingestion_alerts": 0}
    await _run(store, _one_fact_contents(), [golden_fact_recursive()], identity=failing)

    assert len(store.events) == 1  # fact committed; alert loss must not propagate
    assert store.alerts == []


async def test_all_bge_calls_happen_before_the_fact_transaction():
    store = FakeStore()
    client = FakeEmbeddingClient()
    original_embed = client.embed_identity

    async def recording_embed(text, atom_type):
        store.calls.append(("embed", f"/normalize/{atom_type}", (text, atom_type)))
        return await original_embed(text, atom_type)

    client.embed_identity = recording_embed
    await _run(store, _one_fact_contents(), [golden_fact_recursive()], identity=client)

    embed_positions = [index for index, call in enumerate(store.calls) if call[0] == "embed"]
    assert len(embed_positions) == 4
    # The event INSERT is the first statement inside the fact transaction;
    # every bge call must precede it (§3.2.4: HTTP outside the transaction).
    event_insert_position = next(
        index
        for index, call in enumerate(store.calls)
        if call[0] == "fetchrow" and call[1].lstrip().upper().startswith("INSERT") and ".events" in call[1]
    )
    assert max(embed_positions) < event_insert_position


async def test_pg_dimension_reject_rolls_back_whole_fact():
    store = FakeStore()
    wrong_dim = FakeEmbeddingClient(dimension=512)  # fake pgvector column is 1024-wide
    await _run(store, _one_fact_contents(), [golden_fact_recursive()], identity=wrong_dim)

    assert store.events == {}
    assert store.atoms == {}
    assert store.event_atoms == []
    assert store.rollbacks >= 1
    assert len(store.alerts_by_code("event_ingest_failed")) == 1  # §8.4 contract-break path


async def test_alert_details_never_contain_credentials():
    import json as _json

    store = FakeStore()
    failing = FakeEmbeddingClient(identity_failures={("揍", "P"): "transport"})
    await _run(
        store,
        _one_fact_contents(),
        [golden_fact_recursive()],
        identity=failing,
        config=noesis_config(
            noesis_embedding_base_url="http://noesis-user:noesis-password@identity.test"
        ),
    )
    blob = _json.dumps(store.alerts, ensure_ascii=False)
    assert "api_key" not in blob
    assert "Bearer" not in blob
    assert "noesis-user" not in blob
    assert "noesis-password" not in blob
    assert "stack" not in blob.lower()
