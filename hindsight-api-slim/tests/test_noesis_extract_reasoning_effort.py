"""Noesis extraction LLM must send an elevated reasoning_effort for DeepSeek
models.

The extraction client is built through Hyper-Extract's ``create_llm`` with the
retain LLM's provider/model. DeepSeek V4 reasoning models take an explicit
``reasoning_effort`` as a top-level request-body field, forwarded via
ChatOpenAI's ``extra_body``. Elevated reasoning consumes completion budget
before the JSON closure, so the call also raises ``max_tokens`` to keep the
response from being truncated (LengthFinishReasonError). Non-DeepSeek models
must not receive either parameter (they may reject them outright).
"""

from types import SimpleNamespace

import pytest

from hindsight_api.engine.retain import noesis_ingest


def _llm_config(model: str) -> SimpleNamespace:
    return SimpleNamespace(
        provider="openai",
        model=model,
        base_url="https://gateway.example/v1",
        api_key="sk-test",
    )


@pytest.fixture()
def captured_create_llm(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_create_llm(spec, *, api_key="", **kwargs):  # noqa: ANN001, ANN003
        calls.append({"spec": dict(spec), "api_key": api_key, **kwargs})
        return object()

    monkeypatch.setattr("hyperextract.create_llm", fake_create_llm)
    monkeypatch.setattr(
        "hyperextract.noesis.create_noesis_extractor",
        lambda llm_client: object(),
    )
    return calls


def test_deepseek_extraction_sends_configured_reasoning_effort(captured_create_llm):
    factory = noesis_ingest._production_extract_once_factory(
        _llm_config("deepseek-ai/DeepSeek-V4-Flash")
    )

    assert factory is not None
    assert captured_create_llm, "create_llm was not called"
    call = captured_create_llm[0]
    assert call["spec"]["model"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert call["temperature"] == 0
    assert call["extra_body"]["reasoning_effort"] == (
        noesis_ingest.NOESIS_EXTRACT_REASONING_EFFORT
    )
    assert call["extra_body"]["max_tokens"] == noesis_ingest.NOESIS_EXTRACT_MAX_TOKENS


def test_non_deepseek_extraction_does_not_send_reasoning_effort(captured_create_llm):
    factory = noesis_ingest._production_extract_once_factory(_llm_config("gpt-4o-mini"))

    assert factory is not None
    call = captured_create_llm[0]
    extra_body = call.get("extra_body") or {}
    assert "reasoning_effort" not in extra_body
    assert "max_tokens" not in extra_body


def test_multi_member_chain_uses_primary_member_for_reasoning_effort(
    captured_create_llm,
):
    config = SimpleNamespace(
        members=[
            _llm_config("gpt-4o-mini"),
            _llm_config("deepseek-ai/DeepSeek-V4-Flash"),
        ]
    )

    factory = noesis_ingest._production_extract_once_factory(config)

    assert factory is not None
    call = captured_create_llm[0]
    assert call["spec"]["model"] == "gpt-4o-mini"
    extra_body = call.get("extra_body") or {}
    assert "reasoning_effort" not in extra_body
    assert "max_tokens" not in extra_body


def test_hyperextract_create_llm_forwards_extra_body_to_chat_open_ai():
    """Guard the full chain: extra_body reaches ChatOpenAI's request params."""
    from langchain_openai import ChatOpenAI

    from hyperextract import create_llm

    llm = create_llm(
        {"provider": "openai", "model": "deepseek-ai/DeepSeek-V4-Flash"},
        api_key="sk-test",
        temperature=0,
        extra_body={
            "reasoning_effort": noesis_ingest.NOESIS_EXTRACT_REASONING_EFFORT,
            "max_tokens": noesis_ingest.NOESIS_EXTRACT_MAX_TOKENS,
        },
    )
    assert isinstance(llm, ChatOpenAI)
    assert llm.extra_body == {
        "reasoning_effort": noesis_ingest.NOESIS_EXTRACT_REASONING_EFFORT,
        "max_tokens": noesis_ingest.NOESIS_EXTRACT_MAX_TOKENS,
    }
    assert llm._default_params["extra_body"] == llm.extra_body
