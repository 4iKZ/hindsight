"""Read-only Noesis query path: utterance -> single HE call -> filtered flat atoms."""

from types import SimpleNamespace

import pytest

from hindsight_api.engine.retain import noesis_ingest, noesis_query
from hindsight_api.engine.retain.noesis_query import NoesisQueryError, collect_atoms, extract_query_atoms
from tests.noesis_fakes import llm_config, noesis_config


def atom(text, type_="E", role="modifier"):
    return {"text": text, "type": type_, "role": role}


def test_collect_atoms_filters_filler_modifiers_and_dedupes_by_role():
    atoms = collect_atoms(
        [
            atom("订单服务", role="agent"),
            atom("重启", "P", "predicate"),
            atom("后"),
            atom("很"),
            atom("经常"),
            atom("没"),
            atom("订单服务", role="agent"),
            atom("订单服务", role="patient"),
        ]
    )
    assert atoms == [
        atom("订单服务", role="agent"),
        atom("重启", "P", "predicate"),
        atom("经常"),
        atom("没"),
        atom("订单服务", role="patient"),
    ]


def test_collect_atoms_keeps_filler_text_when_not_modifier():
    assert collect_atoms([atom("后", role="agent")]) == [atom("后", role="agent")]


@pytest.fixture
def pipeline(monkeypatch):
    """Seams: the factory carries the query extractor; Hyper-Extract's validator is scripted."""
    import hyperextract.noesis as he_noesis

    state = SimpleNamespace(inputs=[], result=lambda text: [])

    def factory(config, *, query_atoms=False):
        assert query_atoms is True
        extract_once = lambda text: []  # noqa: E731
        extract_once._noesis_query_once = "query-once"
        return extract_once

    def fake_extract(text, *, extract_once):
        assert extract_once == "query-once"
        state.inputs.append(text)
        return state.result(text)

    monkeypatch.setattr(noesis_query, "_production_extract_once_factory", factory)
    monkeypatch.setattr(noesis_query, "redact_document_body", lambda body, config: body)
    monkeypatch.setattr(he_noesis, "extract_noesis_query_atoms", fake_extract)
    return state


async def run(query="为啥连接池被打满？"):
    return await extract_query_atoms(query, llm_config=llm_config(), config=noesis_config())


async def test_returns_filtered_atoms(pipeline):
    pipeline.result = lambda t: [atom("连接池", role="patient"), atom("打满", "P", "predicate"), atom("很")]
    assert await run() == [atom("连接池", role="patient"), atom("打满", "P", "predicate")]
    assert pipeline.inputs == ["为啥连接池被打满？"]


async def test_redaction_runs_before_extraction(pipeline, monkeypatch):
    monkeypatch.setattr(noesis_query, "redact_document_body", lambda body, config: body.replace("secret", "***"))
    await run("password secret")
    assert pipeline.inputs == ["password ***"]


async def test_empty_result_is_empty_not_error(pipeline):
    assert await run("你好，在吗？") == []


async def test_extraction_failure_is_coded(pipeline):
    def boom(text):
        raise RuntimeError("down")

    pipeline.result = boom
    with pytest.raises(NoesisQueryError) as error:
        await run()
    assert error.value.code == "extraction_failed"


async def test_invalid_llm_config_is_coded(monkeypatch):
    def factory(config, *, query_atoms=False):
        raise noesis_ingest.NoesisConfigError("bad")

    monkeypatch.setattr(noesis_query, "_production_extract_once_factory", factory)
    with pytest.raises(NoesisQueryError) as error:
        await run()
    assert error.value.code == "noesis_llm_config_invalid"


def test_factory_attaches_query_extractor_only_on_request(monkeypatch):
    import hyperextract
    import hyperextract.noesis as he_noesis

    client = object()
    seen = []
    monkeypatch.setattr(hyperextract, "create_llm", lambda *args, **kwargs: client)
    monkeypatch.setattr(he_noesis, "create_noesis_extractor", lambda *, llm_client, json_mode=False: lambda text: [])
    monkeypatch.setattr(he_noesis, "create_noesis_summarizer", lambda *, llm_client: lambda text: "s")

    def create_query(*, llm_client, json_mode=False):
        seen.append((llm_client, json_mode))
        return lambda text: {"atoms": []}

    monkeypatch.setattr(he_noesis, "create_noesis_query_extractor", create_query)

    plain = noesis_ingest._production_extract_once_factory(llm_config())
    assert not hasattr(plain, "_noesis_query_once")
    with_query = noesis_ingest._production_extract_once_factory(llm_config(), query_atoms=True)
    assert with_query._noesis_query_once("x") == {"atoms": []}
    assert seen == [(client, False)]
