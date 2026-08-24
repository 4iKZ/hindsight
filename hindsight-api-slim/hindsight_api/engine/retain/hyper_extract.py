"""
Hyper-Extract integration for the retain pipeline.

Runs Hyper-Extract template parsing on retained content in a fire-and-forget
daemon thread, normalizes entities/predicates against a remote GPU embedding
service (bge-m3 @ :8010), and persists triples + hypergraph JSON to a separate
PostgreSQL database (``triple_store`` / ``semantic_event_store`` /
``hypergraph_json_store``).

This module is deliberately isolated from the main retain pipeline: hyper
extraction failures must never affect the retain itself.
"""

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

# Lazy import so a missing hyperextract install degrades gracefully.
try:
    from hyperextract import Template

    HYPER_AVAILABLE = True
except ImportError:
    Template = None
    HYPER_AVAILABLE = False


@dataclass
class _RuntimeConfig:
    """Runtime configuration for the hyper-extraction pipeline.

    Defaults mirror the legacy ``HYPER_*`` environment variables so that direct
    callers (e.g. ``hyper_direct.py``) keep working without a HindsightConfig.
    When a HindsightConfig is available, :func:`_apply_config` overlays its
    ``hyper_*`` / ``norm_*`` fields on top.
    """

    enabled: bool = os.getenv("HYPER_ENABLED", "true").lower() == "true"
    pg_host: str = os.getenv("HYPER_PG_HOST", "172.19.19.26")
    pg_port: int = int(os.getenv("HYPER_PG_PORT", "5432"))
    pg_dbname: str = os.getenv("HYPER_PG_DBNAME", "postgres")
    pg_user: str = os.getenv("HYPER_PG_USER", "postgres")
    pg_password: str | None = os.getenv("HYPER_PG_PASSWORD") or None
    template: str = os.getenv("HYPER_TEMPLATE", "general/biography_graph")
    gpu_norm_url_base: str = os.getenv("HYPER_GPU_NORM_URL_BASE", "http://10.0.0.8:8010")
    norm_threshold_entity: float = 0.70
    norm_threshold_predicate: float = 0.70
    norm_max_aliases_per_canonical: int = 10
    norm_auto_increase_threshold: bool = True
    embedding_dim: int = 1024
    hypergraph_json_file: str = "/tmp/hyper_extract_hypergraph.json"


_runtime = _RuntimeConfig()

# HindsightConfig field name -> _RuntimeConfig attribute.
_CONFIG_FIELD_MAP = {
    "hyper_enabled": "enabled",
    "hyper_pg_host": "pg_host",
    "hyper_pg_port": "pg_port",
    "hyper_pg_dbname": "pg_dbname",
    "hyper_pg_user": "pg_user",
    "hyper_pg_password": "pg_password",
    "hyper_template": "template",
    "hyper_gpu_norm_url_base": "gpu_norm_url_base",
    "hyper_embedding_dim": "embedding_dim",
    "hyper_hypergraph_json_file": "hypergraph_json_file",
    "norm_threshold_entity": "norm_threshold_entity",
    "norm_threshold_predicate": "norm_threshold_predicate",
    "norm_max_aliases_per_canonical": "norm_max_aliases_per_canonical",
    "norm_auto_increase_threshold": "norm_auto_increase_threshold",
}


def _apply_config(config) -> None:
    """Overlay HindsightConfig fields onto the runtime configuration."""
    if config is None:
        return
    for config_field, runtime_attr in _CONFIG_FIELD_MAP.items():
        value = getattr(config, config_field, None)
        if value is not None:
            setattr(_runtime, runtime_attr, value)


# Thread-local storage: one PostgreSQL connection + template per worker thread.
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Structural guard (Step 4)
#
# Purpose: bge-m3 fixes the "true semantics" problem, but the store is also
# polluted by "structural look-alikes" — IPs (172.19.19.117 vs 134), cluster
# nodes (G1N1 vs G1N2), PR numbers (#73780 vs #73770), ISO timestamps, URLs,
# bare numbers, UUIDs. Any embedding model sees such strings as nearly
# identical (very high cosine), so they are recognized here with a regex and
# refused merge — the term becomes its own canonical only.
# ---------------------------------------------------------------------------
STRUCTURAL_TERM_RE = re.compile(
    r"(?i)"
    r"(?:^\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?$)"  # IP[:port]
    r"|(?:^[a-z][a-z0-9+.-]*://\S+$)"  # URL
    r"|(?:^#\d+\b)"  # #PR号 / #issue
    r"|(?:^(?:PR|issue|issues|Merge pull request)[\s#-]*\d+\b)"  # PR #73780 / issue 123 / PR-123
    r"|(?:^g\d+n\d+$)"  # 集群节点 GxNx
    r"|(?:^\d{4}-\d{2}-\d{2}[T ][0-9:.Z+-]+$)"  # ISO 时间戳
    r"|(?:^[0-9]+$)"  # 纯数字
    r"|(?:^\d+\.\d+$)"  # 小数
    r"|(?:^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$)"  # UUID
)


def is_structural_term(term: str) -> bool:
    """Return True when ``term`` matches a structural pattern (IP/node/PR/
    timestamp/URL/number/UUID) that must not be merged with similar terms."""
    if not term or not term.strip():
        return True
    return bool(STRUCTURAL_TERM_RE.match(term.strip()))


def _gpu_norm_url(term_type: str) -> str:
    endpoint = "predicate" if term_type == "predicate" else "entity"
    return f"{_runtime.gpu_norm_url_base}/normalize/{endpoint}"


def normalize_with_gpu(term: str, term_type: str = "predicate", threshold: float = 0.93, conn=None) -> str:
    """Dynamic entity/predicate normalization: exact alias lookup first, then
    GPU embedding service for vector similarity, with structural-term guard and
    alias-count quality gating."""
    if not term or not term.strip():
        return term
    if conn is None:
        return term

    table = "predicate_alias_store" if term_type == "predicate" else "entity_alias_store"
    cur = conn.cursor()

    # 1. Exact alias match
    cur.execute(f"SELECT canonical_name FROM {table} WHERE alias = %s", (term,))
    row = cur.fetchone()
    if row:
        return row[0]

    # 1.5 Structural guard: refuse merge for IP/node/PR/timestamp/URL/number/UUID.
    if is_structural_term(term):
        cur.execute(f"SELECT canonical_name FROM {table} WHERE canonical_name = %s LIMIT 1", (term,))
        exists_struct = cur.fetchone()
        if not exists_struct:
            try:
                # Still try to get a vector for later exact vector retrieval,
                # but the term stays its own canonical.
                resp = httpx.post(
                    _gpu_norm_url(term_type),
                    json={"term": term, "threshold": threshold},
                    timeout=2.0,
                )
                emb = resp.json().get("embedding") if resp.status_code == 200 else None
            except Exception:
                emb = None
            if not emb:
                emb = [0.0] * _runtime.embedding_dim
            cur.execute(
                f"INSERT INTO {table} (canonical_name, alias, alias_embedding) "
                "VALUES (%s, %s, %s::vector) ON CONFLICT (alias) DO NOTHING",
                (term, term, emb),
            )
            conn.commit()
        return term

    # 2. GPU service embedding + vector similarity lookup
    try:
        try:
            resp = httpx.post(
                _gpu_norm_url(term_type),
                json={"term": term, "threshold": threshold},
                timeout=2.0,
            )
        except httpx.TimeoutException:
            logger.warning(f"[HyperExtract] GPU timeout for {term}, retrying...")
            resp = httpx.post(
                _gpu_norm_url(term_type),
                json={"term": term, "threshold": threshold},
                timeout=2.0,
            )

        if resp.status_code == 200:
            data = resp.json()
            emb = data.get("embedding")
            if emb:
                cur.execute(
                    f"SELECT canonical_name, 1 - (alias_embedding <=> %s::vector) as similarity "
                    f"FROM {table} "
                    f"WHERE 1 - (alias_embedding <=> %s::vector) > {threshold} "
                    "ORDER BY similarity DESC LIMIT 1",
                    (emb, emb),
                )
                row = cur.fetchone()
                if row:
                    canonical = row[0]

                    # Quality gating: check alias count for this canonical.
                    cur.execute(
                        f"SELECT count(*) FROM {table} WHERE canonical_name = %s",
                        (canonical,),
                    )
                    alias_count_row = cur.fetchone()
                    alias_count = alias_count_row[0] if alias_count_row else 0

                    # Dynamic threshold: raise the merge bar as a canonical
                    # approaches its alias cap.
                    effective_threshold = threshold
                    if (
                        _runtime.norm_auto_increase_threshold
                        and alias_count >= _runtime.norm_max_aliases_per_canonical * 0.8
                    ):
                        effective_threshold = min(threshold + 0.05, 0.99)
                        logger.info(
                            f"[HyperExtract] Auto-increase threshold for canonical '{canonical}': "
                            f"{threshold} -> {effective_threshold} (aliases={alias_count})"
                        )

                    if alias_count >= _runtime.norm_max_aliases_per_canonical:
                        # Over cap: refuse merge, insert term as a new canonical.
                        logger.warning(
                            f"[HyperExtract] Normalization rejected: canonical '{canonical}' "
                            f"has {alias_count} aliases "
                            f"(max={_runtime.norm_max_aliases_per_canonical}). "
                            f"Term '{term}' will be created as new canonical."
                        )
                        cur.execute(
                            f"INSERT INTO {table} (canonical_name, alias, alias_embedding) "
                            "VALUES (%s, %s, %s::vector) ON CONFLICT (alias) DO NOTHING",
                            (term, term, emb),
                        )
                        conn.commit()
                        return term

                    # Re-check similarity against the effective threshold.
                    if effective_threshold != threshold:
                        cur.execute(
                            f"SELECT canonical_name, 1 - (alias_embedding <=> %s::vector) as similarity "
                            f"FROM {table} "
                            f"WHERE canonical_name = %s "
                            f"AND 1 - (alias_embedding <=> %s::vector) > %s "
                            "ORDER BY similarity DESC LIMIT 1",
                            (emb, canonical, emb, effective_threshold),
                        )
                        row = cur.fetchone()
                        if not row:
                            logger.info(
                                f"[HyperExtract] Normalization rejected by dynamic threshold: "
                                f"term='{term}', canonical='{canonical}', "
                                f"effective_threshold={effective_threshold}"
                            )
                            cur.execute(
                                f"INSERT INTO {table} (canonical_name, alias, alias_embedding) "
                                "VALUES (%s, %s, %s::vector) ON CONFLICT (alias) DO NOTHING",
                                (term, term, emb),
                            )
                            conn.commit()
                            return term

                    cur.execute(
                        f"INSERT INTO {table} (canonical_name, alias, alias_embedding) "
                        "VALUES (%s, %s, %s::vector) ON CONFLICT (alias) DO NOTHING",
                        (canonical, term, emb),
                    )
                    conn.commit()
                    return canonical
                else:
                    cur.execute(
                        f"INSERT INTO {table} (canonical_name, alias, alias_embedding) "
                        "VALUES (%s, %s, %s::vector) ON CONFLICT (alias) DO NOTHING",
                        (term, term, emb),
                    )
                    conn.commit()
                    return term
    except Exception as e:
        logger.warning(f"[HyperExtract] GPU service call failed: {e}")
        cur.execute(
            f"INSERT INTO {table} (canonical_name, alias, alias_embedding) "
            "VALUES (%s, %s, %s::vector) ON CONFLICT (alias) DO NOTHING",
            (term, term, [0.0] * _runtime.embedding_dim),
        )
        conn.commit()
        return term


# ---------------------------------------------------------------------------
# Semantic event dedup
# ---------------------------------------------------------------------------
def get_or_create_semantic_event(canonical_text: str, triples: list, conn) -> tuple:
    """
    Get or create a semantic event.

    Returns: (semantic_event_id, is_new, embedding)
    """
    text_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    cur = conn.cursor()

    # Extract the time range from triples (auxiliary).
    def extract_time_range(triples_list):
        times = []
        for _, _, _, ctx in triples_list:
            t = ctx.get("time")
            if t:
                if isinstance(t, str):
                    try:
                        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
                        times.append(dt)
                    except (ValueError, TypeError):
                        pass
                elif isinstance(t, datetime):
                    times.append(t)
        if times:
            return min(times), max(times)
        return None, None

    def update_time_range(existing_id, db_start, db_end):
        new_start, new_end = extract_time_range(triples)
        if new_start is not None and new_end is not None:
            need_update = False
            if db_start is None or new_start < db_start:
                db_start = new_start
                need_update = True
            if db_end is None or new_end > db_end:
                db_end = new_end
                need_update = True
            if need_update:
                cur.execute(
                    "UPDATE semantic_event_store SET start_time = %s, end_time = %s WHERE id = %s",
                    (db_start, db_end, existing_id),
                )
                conn.commit()

    # 1. Exact hash match
    cur.execute(
        "SELECT id, start_time, end_time FROM semantic_event_store WHERE canonical_hash = %s",
        (text_hash,),
    )
    row = cur.fetchone()
    if row:
        existing_id, db_start, db_end = row
        update_time_range(existing_id, db_start, db_end)
        return existing_id, False, None

    # 2. Vector similarity retrieval
    emb = None
    try:
        resp = httpx.post(
            f"{_runtime.gpu_norm_url_base}/normalize/sentence",
            json={"text": canonical_text},
            timeout=2.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            emb = data.get("embedding")
            if emb:
                cur.execute(
                    "SELECT id, start_time, end_time, 1 - (embedding <=> %s::vector) as similarity "
                    "FROM semantic_event_store "
                    "WHERE 1 - (embedding <=> %s::vector) > 0.99 "
                    "ORDER BY similarity DESC LIMIT 1",
                    (emb, emb),
                )
                row = cur.fetchone()
                if row:
                    existing_id, db_start, db_end, _ = row
                    update_time_range(existing_id, db_start, db_end)
                    return existing_id, False, emb
    except Exception as e:
        logger.warning(f"[HyperExtract] Sentence-BERT service call failed: {e}")

    # 3. Create new event
    event_id = str(uuid.uuid4())
    if emb is None:
        emb = [0.0] * _runtime.embedding_dim
    start_time, end_time = extract_time_range(triples)
    cur.execute(
        "INSERT INTO semantic_event_store "
        "(id, canonical_text, canonical_hash, embedding, triples, start_time, end_time) "
        "VALUES (%s, %s, %s, %s::vector, %s, %s, %s)",
        (event_id, canonical_text, text_hash, emb, json.dumps(triples), start_time, end_time),
    )
    conn.commit()
    return event_id, True, emb


def build_canonical_text_from_triples(triples: list) -> str:
    """
    Build a summary from triples by role + type + order.

    Clause order follows triple insertion order; inside a clause the roles
    fill their positions. Legacy Chinese roles map to the new roles.
    """
    if not triples or not isinstance(triples, list):
        return ""

    # Legacy role -> new role mapping
    ROLE_MAP = {
        "施事": "subj",
        "受事": "obj",
        "使令者": "subj",
        "被使令者": "obj",
        "地点": "loc",
        "时间": "time",
        "工具": "manner",
        "伴随者": "manner",
        "属性": "attr",
        "主题": "attr",
        "值": "attr",
        "原因": "cause",
        "结果": "obj",
    }

    # Group by hyperedge_type, preserving intra-group order.
    events = {}
    for sub, pred, obj, ctx in triples:
        htype = ctx.get("hyperedge_type", "CORE_EVENT")
        events.setdefault(htype, []).append((sub, pred, obj, ctx))

    all_parts = []

    for htype, items in events.items():
        # Build role -> node list
        role_map = {}
        for sub, pred, obj, ctx in items:
            raw_role = ctx.get("role", "")
            role = ROLE_MAP.get(raw_role, raw_role)
            if role:
                role_map.setdefault(role, []).append(sub)

        # Extract the action word
        action = items[0][3].get("action", "") or items[0][1]

        # Generate the clause per type/role
        if htype == "CORE_EVENT":
            subs = role_map.get("subj", [])
            objs = role_map.get("obj", [])
            if subs and action and objs:
                for s in subs:
                    for o in objs:
                        all_parts.append(f"{s} {action} {o}")
            elif subs and action:
                for s in subs:
                    all_parts.append(f"{s} {action}")
            else:
                for sub, pred, obj, ctx in items:
                    all_parts.append(f"{sub} {pred} {obj}")

        elif htype == "MODIFIER":
            locs = role_map.get("loc", [])
            times = role_map.get("time", [])
            for loc in locs:
                all_parts.append(f"在 {loc}")
            for t in times:
                all_parts.append(t)

        elif htype == "CAUSAL":
            causes = role_map.get("cause", [])
            effects = role_map.get("obj", []) or role_map.get("subj", [])
            for c in causes:
                for e in effects:
                    all_parts.append(f"因为 {c}，{e}")

        elif htype == "ATTR_LINK":
            attrs = role_map.get("attr", [])
            targets = role_map.get("obj", []) or role_map.get("subj", [])
            for a in attrs:
                for t in targets:
                    all_parts.append(f"{a}的{t}")

        elif htype == "TEMPORAL":
            times = role_map.get("time", [])
            for t in times:
                all_parts.append(t)

        elif htype == "STATE_LINK":
            states = role_map.get("state", [])
            targets = role_map.get("subj", []) or role_map.get("obj", [])
            for s in states:
                for t in targets:
                    all_parts.append(f"{s}的{t}")

        else:
            # Unknown type: naive join
            for sub, pred, obj, ctx in items:
                if sub and pred and obj:
                    all_parts.append(f"{sub} {pred} {obj}")
                elif sub and obj:
                    all_parts.append(f"{sub} 关联 {obj}")

    # Dedup, preserving order
    seen = set()
    unique_parts = []
    for p in all_parts:
        if p not in seen:
            seen.add(p)
            unique_parts.append(p)

    return "，".join(unique_parts) if unique_parts else "（无有效三元组）"


def _convert_to_triples(result, text, context, conn=None):
    triples = []
    base_context = context.copy()
    base_context["source_text_preview"] = text[:200] + "..." if len(text) > 200 else text

    # One hyperedge_id per retain request, linking all triples of the event.
    hyperedge_id = str(uuid.uuid4())

    if hasattr(result, "edges") and result.edges:
        for edge in result.edges:
            participants = getattr(edge, "participants", None)
            if participants and isinstance(participants, list) and len(participants) > 1:
                # Hyperedge (multiple participants)
                hyperedge_type = getattr(edge, "type", "event")
                roles = getattr(edge, "roles", [])
                if len(roles) < len(participants):
                    roles.extend([""] * (len(participants) - len(roles)))
                edge_name = getattr(edge, "name", "")
                action = getattr(edge, "action", "") or getattr(edge, "name", "") or hyperedge_type
                for idx, participant in enumerate(participants):
                    if not participant:
                        continue
                    ctx = base_context.copy()
                    ctx["hyperedge_id"] = hyperedge_id
                    ctx["hyperedge_type"] = hyperedge_type
                    ctx["role"] = roles[idx] if idx < len(roles) else ""
                    ctx["edge_name"] = edge_name
                    ctx["weight"] = getattr(edge, "confidence", 1.0)
                    ctx["action"] = action
                    edge_time = getattr(edge, "time", None)
                    if edge_time and ("T" in edge_time or ":" in edge_time):
                        ctx["time"] = edge_time
                    else:
                        ctx["time"] = base_context.get("timestamp")
                    if hasattr(edge, "description") and edge.description:
                        ctx["description"] = edge.description
                    participant_norm = normalize_with_gpu(
                        participant, "entity", conn=conn, threshold=_runtime.norm_threshold_entity
                    )
                    triples.append((participant_norm, action, edge_name, ctx))
            else:
                # Plain binary edge
                source = getattr(edge, "source", "")
                target = getattr(edge, "target", "")
                rel_type = getattr(edge, "predicate", None) or getattr(edge, "type", "unknown_relation")
                if source and target:
                    ctx = base_context.copy()
                    ctx["hyperedge_id"] = hyperedge_id
                    edge_time = getattr(edge, "time", None)
                    if edge_time and ("T" in edge_time or ":" in edge_time):
                        ctx["time"] = edge_time
                    else:
                        ctx["time"] = base_context.get("timestamp")
                    if hasattr(edge, "description") and edge.description:
                        ctx["description"] = edge.description
                    ctx["hyperedge_type"] = "binary"
                    ctx["role"] = ""
                    ctx["weight"] = getattr(edge, "confidence", 1.0)
                    source_norm = normalize_with_gpu(
                        source, "entity", conn=conn, threshold=_runtime.norm_threshold_entity
                    )
                    target_norm = normalize_with_gpu(
                        target, "entity", conn=conn, threshold=_runtime.norm_threshold_entity
                    )
                    rel_type_norm = normalize_with_gpu(
                        rel_type, "predicate", conn=conn, threshold=_runtime.norm_threshold_predicate
                    )
                    triples.append((source_norm, rel_type_norm, target_norm, ctx))

    # Semantic dedup + entity normalization
    if conn and triples:
        canonical_text = build_canonical_text_from_triples(triples)
        sem_event_id, is_new, _ = get_or_create_semantic_event(canonical_text, triples, conn)
        occurrence_hyperedge_id = str(uuid.uuid4())

        normalized_triples = []
        for sub, pred, obj, ctx in triples:
            sub_norm = normalize_with_gpu(sub, "entity", conn=conn, threshold=_runtime.norm_threshold_entity)
            obj_norm = normalize_with_gpu(obj, "entity", conn=conn, threshold=_runtime.norm_threshold_entity)
            ctx["semantic_event_id"] = sem_event_id
            ctx["hyperedge_id"] = occurrence_hyperedge_id
            normalized_triples.append((sub_norm, pred, obj_norm, ctx))
        return normalized_triples

    return triples


def _init_hyper_resources():
    """Lazily initialize the PostgreSQL connection and Hyper-Extract template
    (per thread)."""
    if not HYPER_AVAILABLE or not _runtime.enabled:
        return None, None

    try:
        conn = psycopg2.connect(
            host=_runtime.pg_host,
            port=_runtime.pg_port,
            dbname=_runtime.pg_dbname,
            user=_runtime.pg_user,
            password=_runtime.pg_password,
        )
        conn.autocommit = False

        # Create the triple table.
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS triple_store (
                    id BIGSERIAL PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    context JSONB,
                    source_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_triple_subject ON triple_store(subject);
                CREATE INDEX IF NOT EXISTS idx_triple_predicate ON triple_store(predicate);
                CREATE INDEX IF NOT EXISTS idx_triple_object ON triple_store(object);
                """
            )
            # Alias stores used by normalize_with_gpu.
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS entity_alias_store (
                    id BIGSERIAL PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    alias TEXT NOT NULL UNIQUE,
                    alias_embedding VECTOR({_runtime.embedding_dim})
                );
                CREATE TABLE IF NOT EXISTS predicate_alias_store (
                    id BIGSERIAL PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    alias TEXT NOT NULL UNIQUE,
                    alias_embedding VECTOR({_runtime.embedding_dim})
                );
                """
            )
            # Semantic event store (dedup source for canonical sentences).
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS semantic_event_store (
                    id TEXT PRIMARY KEY,
                    canonical_text TEXT,
                    canonical_hash TEXT,
                    embedding VECTOR({_runtime.embedding_dim}),
                    triples JSONB,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP
                );
                """
            )
            conn.commit()

        # Initialize the Hyper-Extract template (custom path or built-in).
        if HYPER_AVAILABLE:
            assert Template is not None
            if _runtime.template.startswith("/") or _runtime.template.startswith("~"):
                template_path = os.path.expanduser(_runtime.template)
                template = Template.create(template_path, language="zh")
            else:
                template = Template.create(_runtime.template, language="zh", extraction_mode="one_stage")
        else:
            template = None

        return conn, template
    except Exception as e:
        logger.error(f"Hyper-Extract init failed: {e}")
        return None, None


def hyper_extract_worker(content: str, bank_id: str):
    """Hyper-Extract worker (runs in a daemon thread)."""
    if not _runtime.enabled:
        logger.debug("[HyperExtract] Disabled by HYPER_ENABLED=false")
        return

    if not HYPER_AVAILABLE:
        logger.warning("[HyperExtract] hyperextract library not installed")
        return

    if not content or not content.strip():
        return

    # Get thread-local resources (connection + template).
    if not hasattr(_thread_local, "hyper_conn") or not hasattr(_thread_local, "hyper_template"):
        conn, template = _init_hyper_resources()
        if conn is None or template is None:
            logger.error("[HyperExtract] Initialization failed")
            return
        _thread_local.hyper_conn = conn
        _thread_local.hyper_template = template

    conn = _thread_local.hyper_conn
    template = _thread_local.hyper_template

    try:
        logger.info(f"[HyperExtract] Parsing text for bank {bank_id} (length={len(content)})")
        result = template.parse(content)

        event_id = str(uuid.uuid4())
        context = {
            "source": "hindsight_retain",
            "bank_id": bank_id,
            "event_id": event_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        triples = _convert_to_triples(result, content, context, conn)

        logger.info(
            f"[HyperExtract] Result nodes: {len(getattr(result, 'nodes', []))}, "
            f"edges: {len(getattr(result, 'edges', []))}"
        )
        logger.info(f"[HyperExtract] Converted {len(triples)} triples")

        # ========== Hypergraph JSON output (file + PostgreSQL) ==========
        # Persist whenever the template parsed nodes/edges (independent of the
        # template path string / type marker).
        try:
            hypergraph_json_file = _runtime.hypergraph_json_file
            nodes = getattr(result, "nodes", [])
            edges = getattr(result, "edges", [])

            # Serialize nodes/edges to dicts.
            nodes_list = []
            for node in nodes:
                if hasattr(node, "dict"):
                    nodes_list.append(node.dict())
                elif hasattr(node, "model_dump"):
                    nodes_list.append(node.model_dump())
                else:
                    nodes_list.append({k: v for k, v in node.__dict__.items() if not k.startswith("_")})

            edges_list = []
            for edge in edges:
                if hasattr(edge, "dict"):
                    edges_list.append(edge.dict())
                elif hasattr(edge, "model_dump"):
                    edges_list.append(edge.model_dump())
                else:
                    edges_list.append({k: v for k, v in edge.__dict__.items() if not k.startswith("_")})

            hypergraph_data = {
                "bank_id": bank_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "nodes": nodes_list,
                "edges": edges_list,
            }

            # Append to file (JSON Lines format, one event per line).
            with open(hypergraph_json_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(hypergraph_data, ensure_ascii=False) + "\n")
            logger.info(f"[HyperExtract] Hypergraph JSON appended to {hypergraph_json_file}")

            # ===== Persist the hypergraph JSON to PostgreSQL =====
            try:
                json_text = json.dumps(hypergraph_data, ensure_ascii=False)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS hypergraph_json_store (
                            id BIGSERIAL PRIMARY KEY,
                            bank_id TEXT,
                            event_id TEXT,
                            nodes JSONB,
                            edges JSONB,
                            full_json JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                    )
                    conn.commit()
                    cur.execute(
                        "INSERT INTO hypergraph_json_store (bank_id, event_id, nodes, edges, full_json) "
                        "VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb)",
                        (
                            bank_id,
                            context.get("event_id"),
                            json.dumps(nodes_list, ensure_ascii=False),
                            json.dumps(edges_list, ensure_ascii=False),
                            json_text,
                        ),
                    )
                    conn.commit()
                logger.info(
                    f"[HyperExtract] Hypergraph JSON inserted into PG hypergraph_json_store "
                    f"(bank={bank_id}, nodes={len(nodes_list)}, edges={len(edges_list)})"
                )
            except Exception as e:
                logger.warning(f"[HyperExtract] Failed to persist hypergraph JSON to PG: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[HyperExtract] Failed to write hypergraph JSON block: {e}")

        if not triples:
            logger.info(f"[HyperExtract] No triples generated for bank {bank_id}")
            return

        # Log a triple summary regardless of debug level.
        logger.info(f"[HyperExtract] Generated {len(triples)} triples for bank {bank_id}")
        for i, (sub, pred, obj, ctx) in enumerate(triples[:5], 1):
            logger.info(f"  Triple {i}: ({sub}, {pred}, {obj}) -> {json.dumps(ctx, ensure_ascii=False)[:200]}")
        if len(triples) > 5:
            logger.info(f"  ... and {len(triples) - 5} more")

        insert_sql = """
            INSERT INTO triple_store (subject, predicate, object, context, source_text)
            VALUES %s
        """
        values = [(sub, pred, obj, json.dumps(ctx), content[:500]) for sub, pred, obj, ctx in triples]

        with conn.cursor() as cur:
            execute_values(cur, insert_sql, values)
            conn.commit()

        logger.info(f"[HyperExtract] Stored {len(triples)} triples for bank {bank_id}")

    except IndexError as e:
        # Hyper-Extract internal merge error (common with long text): skip
        # storage, never affect the main pipeline.
        logger.warning(f"[HyperExtract] IndexError during parsing for bank {bank_id}: {e}")
        return

    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.error(f"[HyperExtract] Error processing for bank {bank_id}: {e}", exc_info=True)

    finally:
        # Close this thread's database connection.
        if hasattr(_thread_local, "hyper_conn") and _thread_local.hyper_conn:
            try:
                _thread_local.hyper_conn.close()
            except Exception:
                pass
            _thread_local.hyper_conn = None


def dispatch_hyper_extract(contents_dicts, bank_id: str, config=None) -> None:
    """Fire-and-forget a hyper-extraction daemon thread for the first content
    item of a retain batch. Failures never affect the retain pipeline."""
    if not contents_dicts:
        return
    if config is not None:
        _apply_config(config)
    content = contents_dicts[0].get("content", "")
    if not content or not content.strip():
        return

    thread = threading.Thread(
        target=hyper_extract_worker,
        args=(content, bank_id),
        daemon=True,
    )
    thread.start()
    logger.info(f"[HyperExtract] Dispatched worker thread for bank {bank_id}")


def reconstruct_sentence_from_sem_event(sem_id: str, conn) -> str:
    """
    Reconstruct a natural-language sentence from a semantic event ID.

    Merges objects sharing the same subject + action into a compact summary.
    """
    cur = conn.cursor()
    cur.execute("SELECT triples FROM semantic_event_store WHERE id = %s", (sem_id,))
    row = cur.fetchone()

    if not row or not row[0]:
        return ""

    triples = row[0]
    if not isinstance(triples, list) or len(triples) == 0:
        return ""

    # Group by (subject, action), collecting all objects.
    groups = {}
    for sub, pred, obj, ctx in triples:
        action = ctx.get("action", "") or pred
        key = (sub, action)
        if key not in groups:
            groups[key] = []
        groups[key].append(obj)

    parts = []
    for (sub, action), objs in groups.items():
        # Dedup objects, preserving order.
        unique_objs = []
        seen_obj = set()
        for obj in objs:
            if obj not in seen_obj:
                seen_obj.add(obj)
                unique_objs.append(obj)

        if len(unique_objs) == 1:
            # Single object: direct join.
            if sub and action and unique_objs[0]:
                parts.append(f"{sub} {action} {unique_objs[0]}")
            elif sub and action:
                parts.append(f"{sub} {action}")
            else:
                parts.append(sub)
        else:
            # Multiple objects joined by 、
            if sub and action:
                parts.append(f"{sub} {action} {'、'.join(unique_objs)}")
            elif sub:
                parts.append(sub)

    # Fallback: join node names.
    if not parts:
        nodes = [item[0] for item in triples if item[0]]
        if nodes:
            parts.append(" ".join(set(nodes)))

    return "，".join(parts) if parts else "（无有效三元组）"
