"""Read-only Noesis recall: atom ids -> abduction engine subprocess -> GLM-polished sentence per rule.

The rule matching and event restoration are the colleague's abduction code (``engines/abduction`` in the
wxs-noesis repo). It is synchronous psycopg2 code with its own venv, so it runs as a short-lived subprocess
(``python -m abduction.recall_cli``: one JSON line in, one JSON line out, read-only DB session). The one
sentence per rule that the engine would get from its own model is produced here with the resolved Noesis LLM
instead; any failure there falls back to the engine's keyword concatenation and is never an error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, ValidationError

from .noesis_ingest import _UNSAFE_PROVIDERS, _resolve_llm_spec, resolve_noesis_llm

logger = logging.getLogger(__name__)

# shortcut: fixed caps; make them config if one deployment needs different numbers.
_MAX_CONCURRENT_ENGINES = 4
_POLISH_TIMEOUT_SECONDS = 20
_STDERR_LOG_CHARS = 500

_engine_slots = asyncio.Semaphore(_MAX_CONCURRENT_ENGINES)

# Copied from abduction/natural_language_rules._llm_text (colleague's prompt); keep in sync.
_POLISH_PROMPT = (
    "你是一个技术表达助手。用户会给你几个关键词（通常是技术术语），"
    "请你把这些关键词组织成一句**简明扼要、逻辑通顺的中文因果描述**，"
    "例如：关键词「冷启动, 缓存, 慢」→ 输出「冷启动后，缓存还未来得及加载，"
    "导致系统运行慢」。要求："
    "1. 一句话讲清因果关系；"
    "2. 自然融入所有关键词，不遗漏；"
    "3. 简明扼要，不要展开、不要写作文、不要加多余修饰。"
)


class NoesisRecallError(Exception):
    """Stable error code for the HTTP layer (502); empty results are NOT errors."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class NoesisRecallUnavailableError(NoesisRecallError):
    """The recall engine is not configured on this deployment (503)."""


class RecallEvent(BaseModel):
    event_id: int = Field(description="Noesis event id of a positive example of the rule.")
    event_time: str | None = Field(default=None, description="Event time, ISO 8601 text.")
    restored: str = Field(default="", description="The event rebuilt from its atoms only.")
    source_text: str = Field(default="", description="Original text stored with the event, if any.")
    tree: list[str] = Field(default_factory=list, description="Predicate hierarchy, one line per predicate.")


class RecallRule(BaseModel):
    rule_id: int
    rule_nl: str = Field(description="One-sentence natural language of the rule.")
    natural_text: str = Field(description="rule_nl followed by the positive/negative example counts.")
    positive_count: int
    negative_count: int
    events: list[RecallEvent] = Field(description="Restored positive examples, at most restore_limit.")


class _EngineRule(RecallRule):
    keywords: list[str] = Field(default_factory=list)


class _EngineOutput(BaseModel):
    rules: list[_EngineRule]


@dataclass(frozen=True)
class NoesisRecallResult:
    atom_ids: list[int]
    rules: list[RecallRule]


def _engine_env(config: Any) -> dict[str, str]:
    """Minimal environment: the child never sees the API's LLM keys or other settings."""
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/root"),
        "LANG": "C.UTF-8",
        "NOESIS_DB_DSN": config.noesis_database_url,
        "NOESIS_DB_SCHEMA": config.noesis_schema,
        "NOESIS_NL_LLM": "off",
    }


async def run_engine(
    atom_ids: list[int], query: str, *, top_n: int, restore_limit: int, config: Any
) -> list[_EngineRule]:
    engine_dir = Path(config.noesis_recall_engine_dir)
    python = engine_dir / ".venv" / "bin" / "python"
    if not python.is_file():
        logger.error("noesis recall engine interpreter not found: %s", python)
        raise NoesisRecallUnavailableError("noesis_recall_unavailable")
    request = json.dumps(
        {"query": query, "atoms": atom_ids, "top_n": top_n, "restore_limit": restore_limit}, ensure_ascii=False
    ).encode()
    async with _engine_slots:
        proc = await asyncio.create_subprocess_exec(
            str(python),
            "-m",
            "abduction.recall_cli",
            cwd=engine_dir,
            env=_engine_env(config),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(request), config.noesis_recall_timeout_seconds)
        except TimeoutError as error:
            proc.kill()
            await proc.wait()
            logger.error("noesis recall engine timed out after %ss", config.noesis_recall_timeout_seconds)
            raise NoesisRecallError("recall_engine_timeout") from error
        except BaseException:
            # client went away (cancelled) or an unexpected error: never leave the child running
            if proc.returncode is None:
                proc.kill()
            raise
    if proc.returncode != 0:
        logger.error(
            "noesis recall engine exited %s: %s", proc.returncode, stderr.decode(errors="replace")[-_STDERR_LOG_CHARS:]
        )
        raise NoesisRecallError("recall_engine_failed")
    lines = [line for line in stdout.decode(errors="replace").splitlines() if line.strip()]
    try:
        return _EngineOutput.model_validate_json(lines[-1]).rules
    except (IndexError, ValidationError) as error:
        logger.error("noesis recall engine output unusable: %s", type(error).__name__)
        raise NoesisRecallError("recall_engine_bad_output") from error


def _with_counts(text: str, positive: int, negative: int) -> str:
    """Same rule as the engine's ``_nl_with_counts``: append the counts unless the text already has them."""
    text = text.strip()
    if text and ("正例" in text or "反例" in text):
        return text
    return f"{text}（正例 {positive} 个，反例 {negative} 个）"


def _build_polish_client(llm_config: Any, config: Any) -> Any | None:
    """Chat client on the resolved Noesis LLM; None (= keyword fallback) when it cannot be built."""
    provider, model, base_url, api_key = _resolve_llm_spec(resolve_noesis_llm(llm_config, config))
    if not provider or not model or provider.lower() in _UNSAFE_PROVIDERS:
        return None
    from hyperextract import create_llm

    spec: dict[str, Any] = {"provider": provider, "model": model, "base_url": base_url}
    # max_tokens covers the reasoning tokens too, so keep it generous.
    options: dict[str, Any] = {"max_tokens": 2048, "timeout": _POLISH_TIMEOUT_SECONDS}
    if urlsplit(base_url).hostname == "open.bigmodel.cn":
        # glm-5.3-flash always thinks and rejects "thinking: disabled" (400, code 1210); low effort ~2s vs ~8s default.
        options["reasoning_effort"] = "low"
    return create_llm(spec, api_key=api_key, temperature=0).bind(**options)


async def _polish_sentence(client: Any | None, keywords: list[str]) -> str | None:
    if client is None or not keywords:
        return None

    def invoke() -> str:
        reply = client.invoke([("system", _POLISH_PROMPT), ("human", f"关键词：{'，'.join(keywords)}")])
        return reply.text.strip()

    try:
        return await asyncio.wait_for(asyncio.to_thread(invoke), _POLISH_TIMEOUT_SECONDS) or None
    except Exception as error:
        logger.warning("noesis recall polish failed (%s), keeping keyword text", type(error).__name__)
        return None


async def polish_rules(rules: list[_EngineRule], *, llm_config: Any, config: Any) -> list[RecallRule]:
    client = None
    if rules:
        try:
            client = _build_polish_client(llm_config, config)
        except Exception as error:
            logger.warning("noesis recall polish client unavailable (%s)", type(error).__name__)
    sentences = await asyncio.gather(*(_polish_sentence(client, rule.keywords) for rule in rules))
    polished: list[RecallRule] = []
    for rule, sentence in zip(rules, sentences, strict=True):
        data = rule.model_dump(exclude={"keywords"})
        if sentence:
            data["rule_nl"] = sentence
            data["natural_text"] = _with_counts(sentence, rule.positive_count, rule.negative_count)
        polished.append(RecallRule.model_validate(data))
    return polished


async def recall_rules(
    atom_ids: list[int], query: str, *, top_n: int, restore_limit: int, llm_config: Any, config: Any
) -> NoesisRecallResult:
    if not atom_ids:
        return NoesisRecallResult(atom_ids=[], rules=[])
    rules = await run_engine(atom_ids, query, top_n=top_n, restore_limit=restore_limit, config=config)
    return NoesisRecallResult(atom_ids=atom_ids, rules=await polish_rules(rules, llm_config=llm_config, config=config))
