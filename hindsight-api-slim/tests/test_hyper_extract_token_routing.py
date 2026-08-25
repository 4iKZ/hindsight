"""Tests for /normalize/token routing in normalize_with_gpu and sentence-context extraction.

The /normalize/token endpoint (context-aware word vectors) is used for entity
normalization when the term literally appears in a local sentence of the source
text. Predicate normalization stays on the sentence-level /normalize/predicate
endpoint (generic-verb over-merge risk), structural terms are guarded by
is_structural_term, and any /normalize/token failure falls back to
/normalize/entity. All HTTP calls are mocked — no network, no database.
"""

import pytest

from hindsight_api.engine.retain import hyper_extract

_EMBEDDING = [0.1] * hyper_extract._runtime.embedding_dim


class _FakeCursor:
    """Cursor stub: exact-alias miss + no similarity row -> term becomes its own canonical."""

    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return None

    def close(self):
        pass


class _FakeConn:
    """Minimal psycopg2-like connection stub."""

    def __init__(self):
        self.cursor_obj = _FakeCursor()
        self.commits = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def close(self):
        pass


def _install_posts(monkeypatch, responses):
    """Install a httpx.post mock that records (url, payload) and replays responses.

    ``responses`` is a list of httpx-style responses consumed in order; the last
    one repeats for any extra calls. Each response is (status_code, json_dict).
    """
    calls = []

    class _Resp:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data

        def json(self):
            return self._data

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        if len(responses) == 1:
            status, data = responses[0]
        else:
            status, data = responses[min(len(calls) - 1, len(responses) - 1)]
        return _Resp(status, data)

    monkeypatch.setattr(hyper_extract.httpx, "post", fake_post)
    return calls


def _token_url():
    return f"{hyper_extract._runtime.gpu_norm_url_base}/normalize/token"


def _entity_url():
    return f"{hyper_extract._runtime.gpu_norm_url_base}/normalize/entity"


def _predicate_url():
    return f"{hyper_extract._runtime.gpu_norm_url_base}/normalize/predicate"


@pytest.fixture(autouse=True)
def _restore_runtime(monkeypatch):
    monkeypatch.setattr(hyper_extract._runtime, "norm_use_context_token", True)


# ---------------------------------------------------------------------------
# _sentence_containing
# ---------------------------------------------------------------------------


class TestSentenceContaining:
    def test_empty_inputs_return_empty(self):
        assert hyper_extract._sentence_containing("", "词") == ""
        assert hyper_extract._sentence_containing("句子", "") == ""

    def test_returns_sentence_containing_term(self):
        text = "节点 G1N1 发生故障。系统其余部分正常。"
        assert hyper_extract._sentence_containing(text, "G1N1") == "节点 G1N1 发生故障。"

    def test_missing_term_returns_empty(self):
        text = "系统运行正常"
        assert hyper_extract._sentence_containing(text, "G1N1") == ""

    def test_multiple_sentences_takes_first_match(self):
        text = "第一句没有它。第二句包含 内存溢出。第三句也有 内存溢出。"
        assert hyper_extract._sentence_containing(text, "内存溢出") == "第二句包含 内存溢出。"

    def test_newline_splits_sentences(self):
        text = "行一 数据库\n行二 数据库"
        assert hyper_extract._sentence_containing(text, "数据库") == "行一 数据库"

    def test_long_sentence_truncates_around_term(self):
        term = "数据库"
        text = ("前" * 400) + term + ("后" * 400)
        got = hyper_extract._sentence_containing(text, term)
        assert len(got) <= 512
        assert term in got
        assert got.startswith("前")

    def test_explicit_max_len_respected(self):
        term = "数据库"
        text = ("前" * 100) + term + ("后" * 100)
        got = hyper_extract._sentence_containing(text, term, max_len=64)
        assert len(got) <= 64
        assert term in got


# ---------------------------------------------------------------------------
# normalize_with_gpu routing
# ---------------------------------------------------------------------------


class TestTokenRouting:
    def test_entity_with_term_in_context_routes_to_token(self, monkeypatch):
        conn = _FakeConn()
        calls = _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        result = hyper_extract.normalize_with_gpu(
            "数据库", "entity", conn=conn, context_sentence="该系统用于管理数据库集群"
        )
        assert calls[0][0] == _token_url()
        assert calls[0][1] == {"word": "数据库", "sentence": "该系统用于管理数据库集群", "threshold": 0.93}
        assert result == "数据库"

    def test_entity_empty_context_uses_entity_endpoint(self, monkeypatch):
        conn = _FakeConn()
        calls = _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        hyper_extract.normalize_with_gpu("数据库", "entity", conn=conn, context_sentence="")
        assert calls[0][0] == _entity_url()

    def test_entity_term_not_in_context_uses_entity_endpoint(self, monkeypatch):
        conn = _FakeConn()
        calls = _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        hyper_extract.normalize_with_gpu(
            "数据集", "entity", conn=conn, context_sentence="用于管理数据库集群"
        )
        assert calls[0][0] == _entity_url()

    def test_predicate_never_routes_to_token(self, monkeypatch):
        conn = _FakeConn()
        calls = _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        hyper_extract.normalize_with_gpu(
            "管理", "predicate", conn=conn, context_sentence="某句包含管理"
        )
        assert calls[0][0] == _predicate_url()

    def test_structural_term_guarded_and_uses_entity_endpoint(self, monkeypatch):
        conn = _FakeConn()
        calls = _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        result = hyper_extract.normalize_with_gpu(
            "G1N1", "entity", conn=conn, context_sentence="节点 G1N1 发生故障"
        )
        assert result == "G1N1"
        assert calls[0][0] == _entity_url()
        assert "/normalize/token" not in calls[0][0]

    def test_token_failure_falls_back_to_entity_endpoint(self, monkeypatch):
        conn = _FakeConn()
        calls = _install_posts(
            monkeypatch,
            [
                (500, {}),  # token endpoint fails
                (200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"}),
            ],
        )
        hyper_extract.normalize_with_gpu(
            "数据库", "entity", conn=conn, context_sentence="该系统用于管理数据库集群"
        )
        assert calls[0][0] == _token_url()
        assert calls[1][0] == _entity_url()

    def test_token_empty_embedding_falls_back_to_entity_endpoint(self, monkeypatch):
        conn = _FakeConn()
        calls = _install_posts(
            monkeypatch,
            [
                (200, {"dim": 1024, "model": "bge-m3"}),  # no embedding key
                (200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"}),
            ],
        )
        hyper_extract.normalize_with_gpu(
            "数据库", "entity", conn=conn, context_sentence="该系统用于管理数据库集群"
        )
        assert calls[0][0] == _token_url()
        assert calls[1][0] == _entity_url()

    def test_disabled_flag_uses_entity_endpoint_even_with_context(self, monkeypatch):
        monkeypatch.setattr(hyper_extract._runtime, "norm_use_context_token", False)
        conn = _FakeConn()
        calls = _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        hyper_extract.normalize_with_gpu(
            "数据库", "entity", conn=conn, context_sentence="该系统用于管理数据库集群"
        )
        assert calls[0][0] == _entity_url()

    def test_no_context_when_conn_none(self, monkeypatch):
        calls = _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        result = hyper_extract.normalize_with_gpu(
            "数据库", "entity", conn=None, context_sentence="该系统用于管理数据库集群"
        )
        assert result == "数据库"
        assert calls == []


# ---------------------------------------------------------------------------
# _CONFIG_FIELD_MAP wiring
# ---------------------------------------------------------------------------


class TestConfigMapping:
    def test_norm_use_context_token_mapped(self):
        assert hyper_extract._CONFIG_FIELD_MAP["norm_use_context_token"] == "norm_use_context_token"

    def test_apply_config_overlays_flag(self, monkeypatch):
        class _DummyConfig:
            norm_use_context_token = False

        monkeypatch.setattr(hyper_extract._runtime, "norm_use_context_token", True)
        hyper_extract._apply_config(_DummyConfig())
        assert hyper_extract._runtime.norm_use_context_token is False
