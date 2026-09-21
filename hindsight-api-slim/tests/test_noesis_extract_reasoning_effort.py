"""Noesis extraction LLM must send reasoning_effort=max for DeepSeek models.

The extraction client is built through Hyper-Extract's ``create_llm`` with the
retain LLM's provider/model. DeepSeek V4 reasoning models take an explicit
``reasoning_effort`` as a top-level request-body field, forwarded via
ChatOpenAI's ``extra_body``. Non-DeepSeek models must not receive the
parameter (they may reject it outright).
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


def test_deepseek_extraction_sends_reasoning_effort_max(captured_create_llm):
    factory = noesis_ingest._production_extract_once_factory(
        _llm_config("deepseek-ai/DeepSeek-V4-Flash")
    )

    assert factory is not None
    assert captured_create_llm, "create_llm was not called"
    call = captured_create_llm[0]
    assert call["spec"]["model"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert call["temperature"] == 0
    assert call["extra_body"]["reasoning_effort"] == "max"


def test_non_deepseek_extraction_does_not_send_reasoning_effort(captured_create_llm):
    factory = noesis_ingest._production_extract_once_factory(_llm_config("gpt-4o-mini"))

    assert factory is not None
    call = captured_create_llm[0]
    assert (call.get("extra_body") or {}).get("reasoning_effort") is None


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
    assert (call.get("extra_body") or {}).get("reasoning_effort") is None


def test_hyperextract_create_llm_forwards_extra_body_to_chat_open_ai():
    """Guard the full chain: extra_body reaches ChatOpenAI's request params."""
    from langchain_openai import ChatOpenAI

    from hyperextract import create_llm

    llm = create_llm(
        {"provider": "openai", "model": "deepseek-ai/DeepSeek-V4-Flash"},
        api_key="sk-test",
        temperature=0,
        extra_body={"reasoning_effort": noesis_ingest.NOESIS_EXTRACT_REASONING_EFFORT},
    )
    assert isinstance(llm, ChatOpenAI)
    assert llm.extra_body == {"reasoning_effort": "max"}
    assert llm._default_params["extra_body"] == {"reasoning_effort": "max"}
