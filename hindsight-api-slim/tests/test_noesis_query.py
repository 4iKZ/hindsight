"""Read-only Noesis query path: utterance -> rewrite -> extraction -> flat atoms."""

from types import SimpleNamespace

import pytest

from hindsight_api.engine.retain import noesis_ingest, noesis_query
from hindsight_api.engine.retain.noesis_query import NoesisQueryError, collect_atoms, extract_query_atoms
from tests.noesis_fakes import llm_config, noesis_config


def atom(text, type_="E", role="modifier"):
    return SimpleNamespace(text=text, type=type_, role=role)


def component(*atoms):
    return SimpleNamespace(atoms=list(atoms))


def outcome(components, alerts=()):
    return SimpleNamespace(components=list(components), alerts=list(alerts), attempts=1)


def test_collect_atoms_filters_filler_modifiers_and_dedupes_by_role():
    atoms = collect_atoms(
        [
            component(
                atom("订单服务", role="agent"),
                atom("重启", "P", "predicate"),
                atom("后"),
                atom("很"),
                atom("经常"),
                atom("没"),
                atom("G1", "G", "agent"),
            ),
            component(atom("订单服务", role="agent"), atom("连接池", role="patient"), atom("订单服务", role="patient")),
        ]
    )
    assert atoms == [
        {"text": "订单服务", "type": "E", "role": "agent"},
        {"text": "重启", "type": "P", "role": "predicate"},
        {"text": "经常", "type": "E", "role": "modifier"},
        {"text": "没", "type": "E", "role": "modifier"},
        {"text": "连接池", "type": "E", "role": "patient"},
        {"text": "订单服务", "type": "E", "role": "patient"},
    ]


def test_collect_atoms_keeps_filler_text_when_not_modifier():
    assert collect_atoms([component(atom("后", role="agent"))]) == [{"text": "后", "type": "E", "role": "agent"}]


@pytest.fixture
def pipeline(monkeypatch):
    """Seams: factory returns an object carrying the rewrite stage; extraction is scripted."""
    import hyperextract.noesis as he_noesis

    state = SimpleNamespace(rewrite_inputs=[], extract_inputs=[], rewrite=lambda t: t, extract=lambda t: outcome([]))

    def rewrite_once(text):
        state.rewrite_inputs.append(text)
        return state.rewrite(text)

    def factory(config, *, query_rewriter=False):
        assert query_rewriter is True
        extract_once = lambda text: []  # noqa: E731
        extract_once._noesis_rewrite_once = rewrite_once
        return extract_once

    def fake_extract(text, *, extract_once, critic_once=None):
        state.extract_inputs.append(text)
        return state.extract(text)

    monkeypatch.setattr(noesis_query, "_production_extract_once_factory", factory)
    monkeypatch.setattr(noesis_query, "redact_document_body", lambda body, config: body)
    monkeypatch.setattr(he_noesis, "extract_noesis_components", fake_extract)
    return state


async def run(query="为啥连接池被打满？"):
    return await extract_query_atoms(query, llm_config=llm_config(), config=noesis_config())


async def test_returns_atoms_from_rewritten_text(pipeline):
    pipeline.rewrite = lambda t: "连接池被打满"
    pipeline.extract = lambda t: outcome([component(atom("连接池", role="agent"), atom("打满", "P", "predicate"))])
    assert await run() == [
        {"text": "连接池", "type": "E", "role": "agent"},
        {"text": "打满", "type": "P", "role": "predicate"},
    ]
    assert pipeline.rewrite_inputs == ["为啥连接池被打满？"]
    assert pipeline.extract_inputs == ["连接池被打满"]


async def test_redaction_runs_before_rewrite(pipeline, monkeypatch):
    monkeypatch.setattr(noesis_query, "redact_document_body", lambda body, config: body.replace("secret", "***"))
    pipeline.rewrite = lambda t: ""
    await run("password secret")
    assert pipeline.rewrite_inputs == ["password ***"]


async def test_empty_rewrite_returns_empty_without_extraction(pipeline):
    pipeline.rewrite = lambda t: ""
    assert await run("你好，在吗？") == []
    assert pipeline.extract_inputs == []


async def test_empty_extraction_is_empty_not_error(pipeline):
    pipeline.rewrite = lambda t: "今天天气"
    assert await run() == []


async def test_rewrite_failure_is_coded_after_one_retry(pipeline):
    def boom(text):
        raise RuntimeError("down")

    pipeline.rewrite = boom
    with pytest.raises(NoesisQueryError) as error:
        await run()
    assert error.value.code == "rewrite_failed"
    assert len(pipeline.rewrite_inputs) == 2


async def test_extraction_crash_is_coded(pipeline):
    pipeline.rewrite = lambda t: "连接池被打满"

    def boom(text):
        raise RuntimeError("down")

    pipeline.extract = boom
    with pytest.raises(NoesisQueryError) as error:
        await run()
    assert error.value.code == "extraction_failed"


async def test_failed_extraction_outcome_is_error_not_empty(pipeline):
    pipeline.rewrite = lambda t: "连接池被打满"
    failed = SimpleNamespace(alert_code="extraction_failed")
    pipeline.extract = lambda t: outcome([], [failed])
    with pytest.raises(NoesisQueryError) as error:
        await run()
    assert error.value.code == "extraction_failed"


async def test_all_components_dropped_is_invalid_not_empty(pipeline):
    pipeline.rewrite = lambda t: "连接池被打满"
    dropped = SimpleNamespace(alert_code="invalid_component_dropped")
    pipeline.extract = lambda t: outcome([], [dropped])
    with pytest.raises(NoesisQueryError) as error:
        await run()
    assert error.value.code == "extraction_invalid"


async def test_partial_drop_returns_remaining_atoms(pipeline):
    pipeline.rewrite = lambda t: "连接池被打满"
    dropped = SimpleNamespace(alert_code="invalid_component_dropped")
    pipeline.extract = lambda t: outcome([component(atom("连接池", role="agent"))], [dropped])
    assert await run() == [{"text": "连接池", "type": "E", "role": "agent"}]


async def test_invalid_llm_config_is_coded(monkeypatch):
    def factory(config, *, query_rewriter=False):
        raise noesis_ingest.NoesisConfigError("bad")

    monkeypatch.setattr(noesis_query, "_production_extract_once_factory", factory)
    with pytest.raises(NoesisQueryError) as error:
        await run()
    assert error.value.code == "noesis_llm_config_invalid"


def test_factory_attaches_rewriter_only_on_request(monkeypatch):
    import hyperextract
    import hyperextract.noesis as he_noesis

    client = object()
    monkeypatch.setattr(hyperextract, "create_llm", lambda *args, **kwargs: client)
    monkeypatch.setattr(he_noesis, "create_noesis_extractor", lambda *, llm_client, json_mode=False: lambda text: [])
    monkeypatch.setattr(he_noesis, "create_noesis_summarizer", lambda *, llm_client: lambda text: "s")
    monkeypatch.setattr(he_noesis, "create_noesis_query_rewriter", lambda *, llm_client: lambda text: f"q:{text}")

    plain = noesis_ingest._production_extract_once_factory(llm_config())
    assert not hasattr(plain, "_noesis_rewrite_once")
    with_rewriter = noesis_ingest._production_extract_once_factory(llm_config(), query_rewriter=True)
    assert with_rewriter._noesis_rewrite_once("x") == "q:x"
