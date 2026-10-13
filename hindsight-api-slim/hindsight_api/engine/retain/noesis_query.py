"""Read-only Noesis query path: one user utterance -> ids of the matching stored atoms.

query -> Memory Defense redaction -> one Hyper-Extract query-atoms call -> filter/dedupe
-> exact (text, atom_type) lookup in ``<schema>.atoms``. Nothing is written and bge is
never called; an atom the store has not seen yet simply yields no id. The LLM is the
Noesis stage's resolved LLM, exactly as in ingestion.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .noesis_ingest import (
    NoesisConfigError,
    _get_pool,
    _production_extract_once_factory,
    resolve_noesis_llm,
    validate_schema_identifier,
)
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


# Exact identity lookup (UNIQUE (text, atom_type)); deprecated atoms are never returned.
_ATOM_IDS_SQL = """
SELECT a.atom_id, a.text, a.atom_type::text AS atom_type
FROM {s}.atoms AS a
JOIN unnest($1::text[], $2::text[]) AS q(text, atom_type)
  ON a.text = q.text AND a.atom_type = q.atom_type::"char"
WHERE a.status = 'A'
"""


async def resolve_atom_ids(atoms: list[dict[str, str]], *, schema: str, pool: Any) -> list[int]:
    """Exact (text, type) -> atom_id, deduped, in the order the atoms were extracted."""
    if not validate_schema_identifier(schema):
        raise NoesisQueryError("noesis_schema_unavailable")
    keys = list(dict.fromkeys((atom["text"], atom["type"]) for atom in atoms))
    if not keys:
        return []
    try:
        rows = await pool.fetch(_ATOM_IDS_SQL.format(s=schema), [k[0] for k in keys], [k[1] for k in keys])
    except Exception as error:
        logger.error("noesis atom id lookup failed: %s", type(error).__name__)
        raise NoesisQueryError("noesis_schema_unavailable") from error
    found = {(row["text"], row["atom_type"]): int(row["atom_id"]) for row in rows}
    return list(dict.fromkeys(found[k] for k in keys if k in found))


async def query_atom_ids(query: str, *, llm_config: Any, config: Any) -> list[int]:
    atoms = await extract_query_atoms(query, llm_config=llm_config, config=config)
    if not atoms:
        return []
    if not getattr(config, "noesis_enabled", False):
        raise NoesisQueryError("noesis_schema_unavailable")
    try:
        pool = await _get_pool(config)
    except Exception as error:
        logger.error("noesis pool unavailable: %s", type(error).__name__)
        raise NoesisQueryError("noesis_schema_unavailable") from error
    return await resolve_atom_ids(atoms, schema=config.noesis_schema, pool=pool)
