"""Tests for predicate normalization in the hyper-extract pipeline.

Predicate normalization (phase-1 leftover fill-in): hyperedge actions and the
second-pass predicate are now normalized through /normalize/predicate
(sentence-level), with a generic-verb blocklist guard (是/有/属于/包含/导致/...)
that refuses merge for high-frequency verbs. Structural terms are guarded by
is_structural_term regardless of type. All HTTP calls are mocked — no network,
no database.
"""

from types import SimpleNamespace

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

    ``responses`` is a list of (status_code, json_dict) consumed in order; the
    last one repeats for any extra calls.
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
        idx = min(len(calls) - 1, len(responses) - 1)
        status, data = responses[idx]
        return _Resp(status, data)

    monkeypatch.setattr(hyper_extract.httpx, "post", fake_post)
    return calls


def _predicate_url():
    return f"{hyper_extract._runtime.gpu_norm_url_base}/normalize/predicate"


# ---------------------------------------------------------------------------
# is_generic_verb
# ---------------------------------------------------------------------------


class TestGenericVerb:
    def test_blocked_verbs(self):
        for v in ("是", "有", "属于", "包含", "导致", "引发", "发生", "使用", "出现", "存在", "达到", "运行"):
            assert hyper_extract.is_generic_verb(v) is True

    def test_specific_verbs_not_blocked(self):
        for v in ("发生故障", "飙升至", "响应变慢", "从...降至...", "部署完成"):
            assert hyper_extract.is_generic_verb(v) is False

    def test_empty_and_none(self):
        assert hyper_extract.is_generic_verb("") is False
        assert hyper_extract.is_generic_verb("  ") is False


# ---------------------------------------------------------------------------
# normalize_with_gpu generic-verb guard
# ---------------------------------------------------------------------------


class TestPredicateGuard:
    def test_generic_verb_predicate_stays_own_canonical(self, monkeypatch):
        conn = _FakeConn()
        calls = _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        result = hyper_extract.normalize_with_gpu("是", "predicate", conn=conn, threshold=0.70)
        assert result == "是"
        # only one call: the predicate endpoint for the new canonical's vector
        assert calls[0][0] == _predicate_url()
        assert calls[0][1] == {"term": "是", "threshold": 0.70}
        # insert happened with term as its own canonical
        insert_params = [params for sql, params in conn.cursor_obj.executed if "INSERT" in str(sql)]
        assert len(insert_params) == 1
        assert insert_params[0][0] == "是"
        assert insert_params[0][1] == "是"

    def test_generic_verb_entity_is_not_guarded(self, monkeypatch):
        conn = _FakeConn()
        _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        # "是" as an entity goes through the normal similarity path (token or
        # entity endpoint), not the generic-verb guard.
        result = hyper_extract.normalize_with_gpu(
            "是", "entity", conn=conn, threshold=0.70, context_sentence="这是主因。"
        )
        assert result == "是"

    def test_structural_predicate_guarded(self, monkeypatch):
        conn = _FakeConn()
        calls = _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        result = hyper_extract.normalize_with_gpu("G1N1", "predicate", conn=conn, threshold=0.70)
        assert result == "G1N1"
        assert calls[0][0] == _predicate_url()


# ---------------------------------------------------------------------------
# _convert_to_triples: hyperedge action + second-pass predicate normalization
# ---------------------------------------------------------------------------


class TestPredicateNormalizationInConvert:
    def _hyperedge_result(self):
        edge = SimpleNamespace(
            participants=["数据库集群主节点", "内存使用率"],
            roles=["subj", "obj"],
            type="CORE_EVENT",
            name="故障",
            action="发生故障",
            time="2026-08-25T04:50:00Z",
            description="数据库集群主节点发生故障",
            confidence=0.9,
        )
        return SimpleNamespace(edges=[edge], nodes=[])

    def _binary_result(self):
        edge = SimpleNamespace(
            source="订单支付服务",
            target="故障发生",
            predicate="从...降至...",
            type="binary",
            time=None,
            description="",
            confidence=0.9,
        )
        return SimpleNamespace(edges=[edge], nodes=[])

    def test_hyperedge_action_normalized(self, monkeypatch):
        conn = _FakeConn()
        _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        result = hyper_extract._convert_to_triples(
            self._hyperedge_result(),
            "数据库集群主节点发生故障，内存使用率飙升。",
            {"source": "test", "bank_id": "b1", "event_id": "e1", "timestamp": "2026-08-25T04:50:00Z"},
            conn,
        )
        # hyperedge triples: (participant_norm, action_norm, edge_name, ctx)
        assert len(result) == 2
        for sub, pred, obj, ctx in result:
            assert pred == "发生故障"
            assert ctx["action"] == "发生故障"

    def test_binary_rel_type_normalized(self, monkeypatch):
        conn = _FakeConn()
        _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        result = hyper_extract._convert_to_triples(
            self._binary_result(),
            "支付成功率从99.5%降至87%。",
            {"source": "test", "bank_id": "b1", "event_id": "e1", "timestamp": "2026-08-25T04:50:00Z"},
            conn,
        )
        assert len(result) == 1
        sub, pred, obj, ctx = result[0]
        assert pred == "从...降至..."

    def test_second_pass_normalizes_predicate_again(self, monkeypatch):
        # The second pass (semantic dedup) re-normalizes sub/pred/obj uniformly.
        # Predicates already normalized in pass 1 short-circuit on exact match,
        # so they are returned unchanged.
        conn = _FakeConn()
        _install_posts(
            monkeypatch,
            [(200, {"embedding": _EMBEDDING, "dim": 1024, "model": "bge-m3"})],
        )
        result = hyper_extract._convert_to_triples(
            self._hyperedge_result(),
            "数据库集群主节点发生故障，内存使用率飙升。",
            {"source": "test", "bank_id": "b1", "event_id": "e1", "timestamp": "2026-08-25T04:50:00Z"},
            conn,
        )
        for sub, pred, obj, ctx in result:
            assert pred == "发生故障"
            assert ctx["semantic_event_id"]
