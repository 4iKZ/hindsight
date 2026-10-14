"""Tests for _strip_code_fences helper in OpenAI-compatible LLM provider."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import (
    OpenAICompatibleLLM,
    _drop_stub_prefix,
    _strip_code_fences,
)


class TestStripCodeFences:
    """Test markdown code fence stripping from LLM responses."""

    def test_bare_json_unchanged(self):
        """Bare JSON passes through unchanged."""
        content = '{"facts": [{"what": "test"}]}'
        assert _strip_code_fences(content) == content

    def test_json_fence_stripped(self):
        """```json ... ``` fences are stripped."""
        content = '```json\n{"facts": [{"what": "test"}]}\n```'
        assert _strip_code_fences(content) == '{"facts": [{"what": "test"}]}'

    def test_plain_fence_stripped(self):
        """``` ... ``` fences without language tag are stripped."""
        content = '```\n{"facts": [{"what": "test"}]}\n```'
        assert _strip_code_fences(content) == '{"facts": [{"what": "test"}]}'

    def test_fence_with_trailing_whitespace(self):
        """Fences with extra whitespace are handled."""
        content = '```json\n{"facts": []}\n```\n'
        result = _strip_code_fences(content)
        assert result == '{"facts": []}'

    def test_fence_with_leading_whitespace(self):
        """Content with leading whitespace before fence."""
        content = '  ```json\n{"facts": []}\n```'
        # The function checks for ``` in content, not startswith
        result = _strip_code_fences(content)
        assert '{"facts": []}' in result

    def test_inner_backticks_preserved(self):
        """Inner triple-backticks inside a JSON string value must not truncate the JSON.

        Regression for the fact-extraction case where an extracted fact describes
        code-fence behavior, so the JSON payload itself contains a literal
        ```` ```json ```` — the old split-based stripper matched that inner
        occurrence and cut the JSON mid-string.
        """
        import json

        content = '```json\n{"facts": [{"what": "the model wraps output in ```json fences"}]}\n```'
        result = _strip_code_fences(content)
        assert result == '{"facts": [{"what": "the model wraps output in ```json fences"}]}'
        parsed = json.loads(result)
        assert parsed["facts"][0]["what"] == "the model wraps output in ```json fences"

    def test_no_fences_no_change(self):
        """Content without any backticks passes through."""
        content = "Just some text without fences"
        assert _strip_code_fences(content) == content

    def test_empty_string(self):
        """Empty string passes through."""
        assert _strip_code_fences("") == ""

    def test_multiline_json(self):
        """Multi-line JSON inside fences is preserved."""
        content = '```json\n{\n  "facts": [\n    {"what": "line1"},\n    {"what": "line2"}\n  ]\n}\n```'
        result = _strip_code_fences(content)
        assert '"line1"' in result
        assert '"line2"' in result
        assert "```" not in result

    def test_missing_closing_fence_recovers_json(self):
        """A fence with no closing ``` still recovers the JSON via the outer-span fallback."""
        content = '```json\n{"facts": []}'
        result = _strip_code_fences(content)
        assert json.loads(result) == {"facts": []}

    def test_prose_wrapped_json_recovered(self):
        """JSON surrounded by prose (no usable fence) is recovered by the fallback."""
        content = 'Sure! Here is the result:\n{"facts": [{"what": "x"}]}\nLet me know if that helps.'
        result = _strip_code_fences(content)
        assert json.loads(result) == {"facts": [{"what": "x"}]}

    def test_non_json_fence_left_for_retry(self):
        """A fenced block that is not JSON yields no valid candidate; content is returned unchanged."""
        content = "```\nnot json at all\n```"
        result = _strip_code_fences(content)
        # No parseable JSON anywhere -> caller sees the stripped body (still a str), never crashes.
        assert isinstance(result, str)
        assert "not json at all" in result

    def test_minimax_style_response(self):
        """Real-world MiniMax response format."""
        content = (
            "```json\n"
            "{\n"
            '  "facts": [\n'
            "    {\n"
            '      "what": "Sebastian switched the Hindsight extraction LLM",\n'
            '      "when": "2026-03-21",\n'
            '      "where": "N/A",\n'
            '      "who": "Sebastian",\n'
            '      "why": "MiniMax wraps JSON in code fences",\n'
            '      "fact_kind": "event",\n'
            '      "fact_type": "world",\n'
            '      "entities": [{"text": "Sebastian"}, {"text": "Hindsight"}],\n'
            '      "labels": {"source_type": "stated", "domain": ["infrastructure"]}\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```"
        )
        result = _strip_code_fences(content)
        assert not result.startswith("```")
        assert not result.endswith("```")
        # Should be valid JSON
        import json

        parsed = json.loads(result)
        assert len(parsed["facts"]) == 1
        assert parsed["facts"][0]["who"] == "Sebastian"


class TestDropStubPrefix:
    """Gateway glitch: a stub of the JSON opening precedes the real, complete object."""

    FACTS = '{"facts": [{"what": "x"}]}'

    @pytest.mark.parametrize("stub", ["{\n", '{\n  "', '{\n  "w', "{ ", '{"', '"facts{ \n'])
    def test_stub_is_dropped(self, stub):
        result = _strip_code_fences(stub + self.FACTS)
        assert json.loads(result) == {"facts": [{"what": "x"}]}

    def test_nested_object_after_stub_is_kept_whole(self):
        content = '{\n{"creates": [], "updates": [{"id": "a", "text": "{b}"}], "deletes": []}'
        assert json.loads(_strip_code_fences(content)) == {
            "creates": [],
            "updates": [{"id": "a", "text": "{b}"}],
            "deletes": [],
        }

    def test_stub_inside_a_code_fence_is_dropped(self):
        result = _strip_code_fences("```json\n{\n" + self.FACTS + "\n```")
        assert json.loads(result) == {"facts": [{"what": "x"}]}

    def test_valid_json_is_never_touched(self):
        assert _strip_code_fences(self.FACTS) == self.FACTS
        assert _drop_stub_prefix(self.FACTS) is None

    def test_prefix_with_colon_is_not_repaired(self):
        """A prefix holding a key/value pair is real content, not a stub: leave it for the retry."""
        content = '{"a": 1, {"facts": []}'
        assert _drop_stub_prefix(content) is None
        assert _strip_code_fences(content) == content

    def test_prefix_length_limit_is_40_chars(self):
        assert _drop_stub_prefix("{" + " " * 39 + self.FACTS) == self.FACTS
        assert _drop_stub_prefix("{" + " " * 40 + self.FACTS) is None

    @pytest.mark.parametrize("content", ['[ {"a": 1}', '[{"a": 1}, {"facts": []}', '{}\n{"facts": []}'])
    def test_prefix_holding_array_or_finished_object_is_not_repaired(self, content):
        """A truncated array or a complete first object must not be 'recovered' by dropping real content."""
        assert _drop_stub_prefix(content) is None

    def test_recovery_is_logged(self, caplog):
        with caplog.at_level("WARNING"):
            _drop_stub_prefix("{\n" + self.FACTS)
        assert "Dropped 2-char garbled prefix" in caplog.text

    def test_truncated_object_is_not_repaired(self):
        content = '{\n{"facts": [{"what": "x"'
        assert _drop_stub_prefix(content) is None
        assert _strip_code_fences(content) == content

    def test_damaged_tail_is_not_repaired(self):
        assert _drop_stub_prefix('{\n{"facts": []} trailing') is None

    def test_non_object_is_not_accepted(self):
        assert _drop_stub_prefix("{\n[1, 2]") is None

    def test_empty_and_plain_text(self):
        assert _drop_stub_prefix("") is None
        assert _drop_stub_prefix("no json here") is None


def _chat_response(content: str) -> MagicMock:
    """Mock OpenAI chat response; see test_openrouter_null_content for why each field is set."""
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = content
    choice.message.tool_calls = None
    choice.message.refusal = None
    response = MagicMock()
    response.error = None
    response.model_dump.return_value = {}
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    response.usage.completion_tokens_details = None
    response.usage.cached_tokens = 0
    response.usage.prompt_tokens_details = None
    response.choices = [choice]
    return response


class _Answer(BaseModel):
    answer: str


class TestCallWithGarbledOpening:
    """End to end through OpenAICompatibleLLM.call: recovery saves the call, unrecoverable output still retries."""

    @staticmethod
    def _llm() -> OpenAICompatibleLLM:
        return OpenAICompatibleLLM(provider="openai", api_key="sk-test", base_url="", model="test-model")

    @pytest.mark.asyncio
    async def test_stub_prefix_is_recovered_without_a_retry(self):
        llm = self._llm()
        with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
            create.return_value = _chat_response('{\n{"answer": "ok"}')
            result = await llm.call(messages=[{"role": "user", "content": "x"}], response_format=_Answer, max_retries=2)
        assert result == _Answer(answer="ok")
        assert create.await_count == 1

    @pytest.mark.asyncio
    async def test_unrecoverable_output_still_uses_the_retry_budget(self):
        llm = self._llm()
        with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
            create.return_value = _chat_response('{"answer": "cut off')
            with pytest.raises(json.JSONDecodeError):
                await llm.call(
                    messages=[{"role": "user", "content": "x"}],
                    response_format=_Answer,
                    max_retries=1,
                    initial_backoff=0.0,
                    max_backoff=0.0,
                )
        assert create.await_count == 2
