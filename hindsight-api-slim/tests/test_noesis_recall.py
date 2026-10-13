"""Noesis recall endpoint: engine subprocess contract, GLM polish fallback, HTTP mapping. Local fakes only."""

import json
import os
import sys
import textwrap
import time
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from hindsight_api.api.http import NoesisRecallRequest, create_app
from hindsight_api.config import HindsightConfig
from hindsight_api.engine.memory_engine import MemoryEngine, NoesisDisabledError
from hindsight_api.engine.retain import noesis_query, noesis_recall
from hindsight_api.engine.retain.noesis_recall import (
    NoesisRecallError,
    NoesisRecallResult,
    NoesisRecallUnavailableError,
    RecallEvent,
    RecallRule,
    _EngineRule,
)
from hindsight_api.extensions import AuthenticationError, OperationValidationError
from hindsight_api.models import RequestContext
from tests.noesis_fakes import llm_config, noesis_config

# A stand-in for engines/abduction: the real command line is `.venv/bin/python -m abduction.recall_cli`,
# and the fake decides what to do from the query text.
FAKE_RECALL_CLI = textwrap.dedent(
    """
    import json, os, sys, time

    req = json.load(sys.stdin)
    q = req["query"]
    rule = {"rule_id": 7, "rule_nl": "甲，乙", "natural_text": "甲，乙（正例 2 个，反例 0 个）",
            "positive_count": 2, "negative_count": 0, "keywords": ["甲", "乙"],
            "events": [{"event_id": 169, "event_time": "2027-09-30", "restored": "还原。",
                        "source_text": "原文", "tree": ["行"]}]}
    if q == "fail":
        sys.stderr.write("boom")
        sys.exit(3)
    if q == "junk":
        print("not json")
    elif q == "slow":
        time.sleep(30)
    elif q == "noisy":
        print("stray line before the result")
        print(json.dumps({"rules": [rule]}))
    elif q == "env":
        rule["rule_nl"] = json.dumps({"keys": sorted(os.environ), "dsn": os.environ["NOESIS_DB_DSN"],
                                      "schema": os.environ["NOESIS_DB_SCHEMA"], "nl": os.environ["NOESIS_NL_LLM"],
                                      "cwd": os.getcwd(), "req": req})
        print(json.dumps({"rules": [rule]}))
    else:
        print(json.dumps({"rules": [rule]}))
    """
)


@pytest.fixture
def engine_dir(tmp_path):
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").symlink_to(sys.executable)
    package = tmp_path / "abduction"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "recall_cli.py").write_text(FAKE_RECALL_CLI)
    return tmp_path


def recall_config(engine_dir, **overrides):
    settings = {
        "noesis_recall_engine_dir": str(engine_dir),
        "noesis_recall_timeout_seconds": 20,
        "noesis_database_url": "postgresql://secret-user:secret-pw@db.test:5432/noesis",
    }
    return noesis_config(**(settings | overrides))


async def run(engine_dir, query="ok", **overrides):
    return await noesis_recall.run_engine(
        [3, 5], query, top_n=3, restore_limit=5, config=recall_config(engine_dir, **overrides)
    )


async def test_engine_success_parses_rules(engine_dir):
    (rule,) = await run(engine_dir)
    assert (rule.rule_id, rule.positive_count, rule.negative_count) == (7, 2, 0)
    assert rule.keywords == ["甲", "乙"]
    assert rule.events == [
        RecallEvent(event_id=169, event_time="2027-09-30", restored="还原。", source_text="原文", tree=["行"])
    ]


async def test_engine_tolerates_stray_stdout_lines(engine_dir):
    (rule,) = await run(engine_dir, "noisy")
    assert rule.rule_id == 7


async def test_engine_gets_minimal_env_and_request(engine_dir, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-must-not-leak")
    (rule,) = await run(engine_dir, "env", noesis_schema="s1")
    seen = json.loads(rule.rule_nl)
    assert "HINDSIGHT_API_LLM_API_KEY" not in seen["keys"]
    assert seen["dsn"] == "postgresql://secret-user:secret-pw@db.test:5432/noesis"
    assert (seen["schema"], seen["nl"]) == ("s1", "off")
    assert seen["cwd"] == str(engine_dir)
    assert seen["req"] == {"query": "env", "atoms": [3, 5], "top_n": 3, "restore_limit": 5}


@pytest.mark.parametrize("query,code", [("fail", "recall_engine_failed"), ("junk", "recall_engine_bad_output")])
async def test_engine_failures_map_to_codes(engine_dir, query, code):
    with pytest.raises(NoesisRecallError) as error:
        await run(engine_dir, query)
    assert error.value.code == code


async def test_engine_timeout_kills_child(engine_dir):
    started = time.monotonic()
    with pytest.raises(NoesisRecallError) as error:
        await run(engine_dir, "slow", noesis_recall_timeout_seconds=1)
    assert error.value.code == "recall_engine_timeout"
    assert time.monotonic() - started < 10  # the child sleeps 30 s: it was killed, not awaited


async def test_engine_missing_interpreter_is_unavailable(tmp_path):
    with pytest.raises(NoesisRecallUnavailableError) as error:
        await run(tmp_path)
    assert error.value.code == "noesis_recall_unavailable"


# ---------------------------------------------------------------------------
# Polish
# ---------------------------------------------------------------------------


def engine_rule(**overrides):
    data = {
        "rule_id": 7,
        "rule_nl": "甲，乙",
        "natural_text": "甲，乙（正例 2 个，反例 1 个）",
        "positive_count": 2,
        "negative_count": 1,
        "keywords": ["甲", "乙"],
        "events": [],
    }
    return _EngineRule.model_validate(data | overrides)


class FakeClient:
    def __init__(self, reply="甲导致乙。", error=None, delay=0.0):
        self.reply, self.error, self.delay, self.calls = reply, error, delay, []

    def invoke(self, messages):
        self.calls.append(messages)
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return SimpleNamespace(text=f" {self.reply} ")


async def polish(monkeypatch, rules, client):
    monkeypatch.setattr(noesis_recall, "_build_polish_client", lambda llm, cfg: client)
    return await noesis_recall.polish_rules(rules, llm_config=llm_config(), config=noesis_config())


async def test_polish_replaces_sentence_and_adds_counts(monkeypatch):
    client = FakeClient()
    (rule,) = await polish(monkeypatch, [engine_rule()], client)
    assert rule.rule_nl == "甲导致乙。"
    assert rule.natural_text == "甲导致乙。（正例 2 个，反例 1 个）"
    assert not hasattr(rule, "keywords")
    system, human = client.calls[0]
    assert system[0] == "system" and "技术表达助手" in system[1]
    assert human == ("human", "关键词：甲，乙")


async def test_polish_does_not_duplicate_counts(monkeypatch):
    (rule,) = await polish(monkeypatch, [engine_rule()], FakeClient("共 2 个正例，1 个反例"))
    assert rule.natural_text == "共 2 个正例，1 个反例"


@pytest.mark.parametrize(
    "client", [FakeClient(error=RuntimeError("down")), FakeClient(reply=""), None], ids=["error", "empty", "no-client"]
)
async def test_polish_failure_keeps_keyword_text(monkeypatch, client):
    (rule,) = await polish(monkeypatch, [engine_rule()], client)
    assert (rule.rule_nl, rule.natural_text) == ("甲，乙", "甲，乙（正例 2 个，反例 1 个）")


async def test_polish_timeout_keeps_keyword_text(monkeypatch):
    monkeypatch.setattr(noesis_recall, "_POLISH_TIMEOUT_SECONDS", 0.2)
    (rule,) = await polish(monkeypatch, [engine_rule()], FakeClient(delay=1.0))
    assert rule.rule_nl == "甲，乙"


async def test_polish_skips_rules_without_keywords_and_is_per_rule(monkeypatch):
    client = FakeClient()
    rules = [engine_rule(rule_id=1, keywords=[]), engine_rule(rule_id=2)]
    first, second = await polish(monkeypatch, rules, client)
    assert (first.rule_nl, second.rule_nl) == ("甲，乙", "甲导致乙。")
    assert len(client.calls) == 1


async def test_polish_with_no_rules_builds_no_client(monkeypatch):
    build = AsyncMock(side_effect=AssertionError("no client without rules"))
    monkeypatch.setattr(noesis_recall, "_build_polish_client", build)
    assert await noesis_recall.polish_rules([], llm_config=llm_config(), config=noesis_config()) == []


def install_fake_hyperextract(monkeypatch):
    captured = {}

    class Client:
        def bind(self, **kwargs):
            captured["bind"] = kwargs
            return self

    def create_llm(spec, **kwargs):
        captured.update(spec=spec, kwargs=kwargs)
        return Client()

    module = ModuleType("hyperextract")
    module.create_llm = create_llm
    monkeypatch.setitem(sys.modules, "hyperextract", module)
    return captured


def glm_llm():
    return SimpleNamespace(
        provider="openai", model="glm-5.3-flash", base_url="https://open.bigmodel.cn/api/paas/v4", api_key="k"
    )


def test_polish_client_disables_thinking_on_glm(monkeypatch):
    captured = install_fake_hyperextract(monkeypatch)
    assert noesis_recall._build_polish_client(glm_llm(), noesis_config()) is not None
    assert captured["spec"]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["kwargs"] == {"api_key": "k", "temperature": 0}
    assert captured["bind"]["max_tokens"] == 256


def test_polish_client_leaves_other_providers_alone(monkeypatch):
    captured = install_fake_hyperextract(monkeypatch)
    assert noesis_recall._build_polish_client(llm_config(), noesis_config()) is not None
    assert "extra_body" not in captured["spec"]


@pytest.mark.parametrize("llm", [llm_config(provider="gemini"), llm_config(model="")])
def test_polish_client_none_when_unusable(monkeypatch, llm):
    install_fake_hyperextract(monkeypatch)
    assert noesis_recall._build_polish_client(llm, noesis_config()) is None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def test_recall_rules_without_atoms_never_starts_engine(monkeypatch):
    monkeypatch.setattr(noesis_recall, "run_engine", AsyncMock(side_effect=AssertionError("no atoms, no engine")))
    result = await noesis_recall.recall_rules(
        [], "q", top_n=3, restore_limit=5, llm_config=llm_config(), config=noesis_config()
    )
    assert result == NoesisRecallResult(atom_ids=[], rules=[])


async def test_recall_rules_runs_engine_then_polish(engine_dir, monkeypatch):
    monkeypatch.setattr(noesis_recall, "_build_polish_client", lambda llm, cfg: FakeClient())
    result = await noesis_recall.recall_rules(
        [3, 5], "ok", top_n=3, restore_limit=5, llm_config=llm_config(), config=recall_config(engine_dir)
    )
    assert result.atom_ids == [3, 5]
    assert [rule.rule_nl for rule in result.rules] == ["甲导致乙。"]


# ---------------------------------------------------------------------------
# Engine method
# ---------------------------------------------------------------------------


def make_engine(*, enabled=True, engine_dir="/opt/abduction"):
    engine = object.__new__(MemoryEngine)
    engine._authenticate_tenant = AsyncMock()
    engine._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(
            return_value=replace(
                HindsightConfig.from_env(), noesis_enabled=enabled, noesis_recall_engine_dir=engine_dir
            )
        )
    )
    engine._retain_llm_config = SimpleNamespace(with_config=lambda *a, **k: llm_config())
    return engine


async def test_engine_method_flow(monkeypatch):
    atoms = AsyncMock(return_value=[3, 5])
    recall = AsyncMock(return_value=NoesisRecallResult(atom_ids=[3, 5], rules=[]))
    monkeypatch.setattr(noesis_query, "query_atom_ids", atoms)
    monkeypatch.setattr(noesis_recall, "recall_rules", recall)
    engine = make_engine()
    result = await engine.recall_noesis("bank", "q", top_n=2, restore_limit=1, request_context=RequestContext())
    assert result.atom_ids == [3, 5]
    engine._authenticate_tenant.assert_awaited_once()
    assert recall.call_args.args == ([3, 5], "q")
    assert (recall.call_args.kwargs["top_n"], recall.call_args.kwargs["restore_limit"]) == (2, 1)


@pytest.mark.parametrize(
    "kwargs,error", [({"enabled": False}, NoesisDisabledError), ({"engine_dir": ""}, NoesisRecallUnavailableError)]
)
async def test_engine_method_gates_before_any_llm_call(monkeypatch, kwargs, error):
    monkeypatch.setattr(noesis_query, "query_atom_ids", AsyncMock(side_effect=AssertionError("gated first")))
    with pytest.raises(error):
        await make_engine(**kwargs).recall_noesis(
            "bank", "q", top_n=3, restore_limit=5, request_context=RequestContext()
        )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_request_defaults_and_bounds():
    body = NoesisRecallRequest(query="磁盘写满")
    assert (body.top_n, body.restore_limit) == (3, 5)
    for bad in (
        {"query": " "},
        {"query": "x" * 2001},
        {"query": "q", "top_n": 0},
        {"query": "q", "top_n": 11},
        {"query": "q", "restore_limit": -1},
        {"query": "q", "restore_limit": 21},
    ):
        with pytest.raises(ValueError):
            NoesisRecallRequest(**bad)


@pytest.fixture
def http_engine():
    return SimpleNamespace(audit_logger=None, _audit_logger=None, _operation_validator=None, recall_noesis=AsyncMock())


async def http_post(engine, body):
    app = create_app(engine, initialize_memory=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/v1/default/banks/bank/noesis/recall", json=body)


async def test_http_success_shape(http_engine):
    rule = RecallRule(
        rule_id=7,
        rule_nl="甲导致乙。",
        natural_text="甲导致乙。（正例 2 个，反例 0 个）",
        positive_count=2,
        negative_count=0,
        events=[RecallEvent(event_id=169, event_time="2027-09-30", restored="还原。", source_text="原文", tree=["行"])],
    )
    http_engine.recall_noesis.return_value = NoesisRecallResult(atom_ids=[3, 5], rules=[rule])
    response = await http_post(http_engine, {"query": "磁盘写满", "top_n": 2, "restore_limit": 1})
    assert response.status_code == 200
    assert response.json() == {
        "query": "磁盘写满",
        "count": 1,
        "atom_ids": [3, 5],
        "rules": [rule.model_dump()],
    }
    assert http_engine.recall_noesis.call_args.kwargs["top_n"] == 2
    assert http_engine.recall_noesis.call_args.kwargs["restore_limit"] == 1


async def test_http_no_match_is_200_with_empty_rules(http_engine):
    http_engine.recall_noesis.return_value = NoesisRecallResult(atom_ids=[], rules=[])
    response = await http_post(http_engine, {"query": "你好"})
    assert response.status_code == 200
    assert response.json() == {"query": "你好", "count": 0, "atom_ids": [], "rules": []}


@pytest.mark.parametrize(
    "error,status,detail",
    [
        (NoesisDisabledError(), 503, {"error": "noesis_disabled"}),
        (NoesisRecallUnavailableError("noesis_recall_unavailable"), 503, {"error": "noesis_recall_unavailable"}),
        (noesis_query.NoesisQueryError("extraction_failed"), 502, {"error": "extraction_failed"}),
        (NoesisRecallError("recall_engine_timeout"), 502, {"error": "recall_engine_timeout"}),
        (OperationValidationError("quota", 402), 402, "quota"),
        (AuthenticationError("denied"), 401, None),
        (RuntimeError("internal"), 500, "internal"),
    ],
)
async def test_http_error_mapping(http_engine, error, status, detail):
    http_engine.recall_noesis.side_effect = error
    response = await http_post(http_engine, {"query": "q"})
    assert response.status_code == status
    if detail is not None:
        assert response.json()["detail"] == detail


@pytest.mark.parametrize("body", [{}, {"query": " "}, {"query": "q", "top_n": 0}, {"query": "q", "restore_limit": 99}])
async def test_http_validation_does_not_call_engine(http_engine, body):
    response = await http_post(http_engine, body)
    assert response.status_code == 422
    http_engine.recall_noesis.assert_not_awaited()


def test_recall_settings_are_static_config():
    configurable = HindsightConfig.get_configurable_fields()
    assert "noesis_recall_engine_dir" not in configurable
    assert "noesis_recall_timeout_seconds" not in configurable
    assert os.environ.get("HINDSIGHT_API_NOESIS_RECALL_ENGINE_DIR") is None
