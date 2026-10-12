"""Read-only Noesis query path: one user utterance -> flat list of main atoms.

query -> Memory Defense redaction -> one Hyper-Extract query-atoms call -> filter/dedupe
(no DB, no vectors, nothing is written). The LLM is the bank's resolved retain LLM,
exactly as in ingestion.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .noesis_ingest import NoesisConfigError, _production_extract_once_factory, resolve_noesis_llm
from .orchestrator import redact_document_body

logger = logging.getLogger(__name__)

# shortcut: backstop for connectors / degree words the prompt already tells the model to omit;
# extend when new filler shows up in modifier atoms. Only role=modifier is filtered.
_FUNCTION_WORDS = frozenset(
    {"后", "之后", "以后", "所以", "因为", "因", "因此", "并", "且", "但", "而", "然后", "于是", "接着", "随后"}
    | {"很", "非常", "比较", "特别"}
)


class NoesisQueryError(Exception):
    """Stable error code for the HTTP layer; empty results are NOT errors."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def collect_atoms(atoms: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop filler modifiers and dedupe by (text, type, role), keeping first-seen order."""
    seen: set[tuple[str, str, str]] = set()
    kept: list[dict[str, str]] = []
    for atom in atoms:
        key = (atom["text"], atom["type"], atom["role"])
        if (atom["role"] == "modifier" and atom["text"] in _FUNCTION_WORDS) or key in seen:
            continue
        seen.add(key)
        kept.append(atom)
    return kept


async def extract_query_atoms(query: str, *, llm_config: Any, config: Any) -> list[dict[str, str]]:
    try:
        from hyperextract.noesis import extract_noesis_query_atoms
    except ImportError as error:
        raise NoesisQueryError("noesis_llm_config_invalid") from error
    try:
        extract_once = _production_extract_once_factory(resolve_noesis_llm(llm_config, config), query_atoms=True)
    except NoesisConfigError as error:
        logger.error("noesis query config invalid: %s", error)
        raise NoesisQueryError("noesis_llm_config_invalid") from error

    text = redact_document_body(query, config)
    try:
        atoms = await asyncio.to_thread(extract_noesis_query_atoms, text, extract_once=extract_once._noesis_query_once)
    except Exception as error:
        logger.error("noesis query extraction failed: %s", type(error).__name__)
        raise NoesisQueryError("extraction_failed") from error
    return collect_atoms(atoms)
