"""Read-only Noesis query path: one user utterance -> flat list of main atoms.

query -> Memory Defense redaction -> rewrite into statements -> Hyper-Extract closure
extraction -> collect atoms (no DB, no vectors, nothing is written). The extraction LLM
is the bank's resolved retain LLM, exactly as in ingestion.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .noesis_ingest import NoesisConfigError, _production_extract_once_factory
from .orchestrator import redact_document_body

logger = logging.getLogger(__name__)

# shortcut: hand-written list from the first real runs; extend when new filler shows up in
# modifier atoms. Only role=modifier is filtered, so agent/patient/predicate always pass.
_FUNCTION_WORDS = frozenset(
    {"后", "之后", "以后", "所以", "因为", "因", "因此", "并", "且", "但", "而", "然后", "于是", "接着", "随后"}
    | {"很", "非常", "比较", "特别"}
)


class NoesisQueryError(Exception):
    """Stable error code for the HTTP layer; empty results are NOT errors."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def collect_atoms(components: Any) -> list[dict[str, str]]:
    """Flatten fact/hypothesis atoms: drop system G atoms and filler modifiers, dedupe by (text, type, role)."""
    seen: set[tuple[str, str, str]] = set()
    atoms: list[dict[str, str]] = []
    for component in components:
        for atom in component.atoms:
            if atom.type == "G" or (atom.role == "modifier" and atom.text in _FUNCTION_WORDS):
                continue
            key = (atom.text, atom.type, atom.role)
            if key not in seen:
                seen.add(key)
                atoms.append({"text": atom.text, "type": atom.type, "role": atom.role})
    return atoms


async def extract_query_atoms(query: str, *, llm_config: Any, config: Any) -> list[dict[str, str]]:
    try:
        from hyperextract.noesis import extract_noesis_components, rewrite_noesis_query
    except ImportError as error:
        raise NoesisQueryError("noesis_llm_config_invalid") from error
    try:
        extract_once = _production_extract_once_factory(llm_config, query_rewriter=True)
    except NoesisConfigError as error:
        logger.error("noesis query config invalid: %s", error)
        raise NoesisQueryError("noesis_llm_config_invalid") from error

    text = redact_document_body(query, config)
    try:
        rewritten = await asyncio.to_thread(rewrite_noesis_query, text, rewrite_once=extract_once._noesis_rewrite_once)
    except Exception as error:
        logger.error("noesis query rewrite failed: %s", type(error).__name__)
        raise NoesisQueryError("rewrite_failed") from error
    if not rewritten:
        return []
    try:
        outcome = await asyncio.to_thread(extract_noesis_components, rewritten, extract_once=extract_once)
    except Exception as error:
        logger.error("noesis query extraction failed: %s", type(error).__name__)
        raise NoesisQueryError("extraction_failed") from error
    if not outcome.components and any(a.alert_code == "invalid_component_dropped" for a in outcome.alerts):
        raise NoesisQueryError("extraction_invalid")
    return collect_atoms(outcome.components)
