"""Noesis shared bge embedding client (requirements 03 + 05).

One client, one HTTP implementation, two vector kinds:

* ``embed_identity(text, atom_type)`` — the write-only identity vectors of
  requirement 03: ``BGE(pure literal)`` — the atom text itself, no context,
  no prefixes, no role/type concatenation. Endpoint mapping frozen
  (requirement 03 §7.3): ``E`` → ``/normalize/entity``, ``P`` →
  ``/normalize/predicate``, body ``{"term": text}``.
* ``embed_context(context_text)`` — the throwaway predicate-frame context
  vectors of requirement 05 (§7): ``POST /normalize/sentence`` with body
  exactly ``{"text": context_text}``, used for anchor routing only and never
  persisted.

``/normalize/token`` is never used (it returns a contextual hidden-state
pooling, not an identity vector). Both public methods share one
``httpx.AsyncClient``, one ``/health`` gate, and one retry + validation
ladder (private ``_post_and_validate``) — no second HTTP implementation.

Failure semantics (requirement 03 §7.4/§7.5/§9, reused verbatim by
requirement 05 §7.5): every call is encoded on its own, transport errors
and 5xx are retried up to ``max_retries``, 4xx/422 are never retried, a
wrong response dimension is a per-call failure, and there is no fallback —
no zero vectors, no random vectors, no local model. The first use per
process runs ``GET /health``; config-level mismatches (dim/model/status)
disable vectors for the process (cached), transport failures are not cached
and the next batch retries.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Frozen endpoint map (requirement 03 §7.3). G never gets a vector.
_IDENTITY_ENDPOINTS = {"E": "/normalize/entity", "P": "/normalize/predicate"}

# Frozen context endpoint (requirement 05 §7.2/§7.3).
_CONTEXT_ENDPOINT = "/normalize/sentence"

# error_kind values recorded in identity_vector / anchor_context alert details.
_TRANSPORT_KINDS = frozenset({"timeout", "transport", "http_5xx"})


def normalize_model_basename(model: str) -> str:
    """Lowercase basename after the last '/': ``BAAI/bge-m3`` ≡ ``bge-m3``.

    Exact equality after normalization — never substring/contains, so a
    different model like ``bge-m3-evil`` can never pass the gate.
    """
    return model.strip().lower().rsplit("/", 1)[-1]


def is_transport_kind(error_kind: str) -> bool:
    """True for error kinds that mean 'service unavailable' when universal."""
    return error_kind in _TRANSPORT_KINDS


class EmbeddingConfigInvalid(Exception):
    """The service /health contradicts the configured generation.

    Process-level disable (cached on the client instance). Since requirement 05
    made Context a hard precondition, a component that hits this drops whole
    with the ``anchor_context_config_invalid`` alert instead of degrading to
    NULL identity vectors.
    """


class EmbeddingServiceUnavailable(Exception):
    """Transport-level /health failure (never cached; next batch retries)."""


class EmbeddingError(Exception):
    """A single literal or context could not be embedded; it registers with
    a NULL vector (identity) or aborts the component (context)."""

    def __init__(self, error_kind: str, message: str) -> None:
        super().__init__(message)
        self.error_kind = error_kind


class EmbeddingProfileUnavailable(Exception):
    """The shared E/P embedding-space generation gate refused this component.

    Carries the fixed gate ``reason`` and the sanitized db/config profile
    snapshots (requirement 05A §5.2/§5.3). Raised before any Context or
    Identity HTTP call, or by the in-transaction FOR SHARE fence re-check.
    """

    def __init__(
        self,
        reason: str,
        *,
        db_profile: dict[str, Any] | None = None,
        config_profile: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.db_profile = db_profile
        self.config_profile = config_profile


class NoesisEmbeddingClient:
    """Shared per-process bge client (lazy singleton, closed with the pool).

    ``http_client`` is the test seam (``httpx.MockTransport``); production
    builds one ``httpx.AsyncClient`` bound to ``base_url``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        revision: str,
        dimension: int,
        timeout_seconds: float,
        max_retries: int,
        api_key: str = "",
        http_client: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.revision = revision
        self.dimension = dimension
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        # Sent per-request so the injected http_client test seam exhibits the
        # same auth behavior as the owned production client.
        self._auth_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = http_client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout_seconds)
        self._owns_http = http_client is None
        # None = unchecked, "ready", "config_invalid". Transport failures are
        # deliberately never written here, so the next batch re-probes.
        self._health_state: str | None = None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # /health gate (requirement 03 §7.4)
    # ------------------------------------------------------------------

    async def ensure_ready(self) -> None:
        """Verify the live service matches the configured generation, once."""
        if self._health_state == "ready":
            return
        if self._health_state == "config_invalid":
            raise EmbeddingConfigInvalid("embedding service /health failed config validation (cached)")
        try:
            response = await self._http.get("/health", headers=self._auth_headers)
        except httpx.TransportError as error:
            raise EmbeddingServiceUnavailable(f"/health transport failure: {type(error).__name__}") from error
        if response.status_code != 200:
            # A sick service may recover; do not cache, do not guess.
            raise EmbeddingServiceUnavailable(f"/health answered HTTP {response.status_code}")
        try:
            payload = response.json()
        except Exception as error:
            self._health_state = "config_invalid"
            raise EmbeddingConfigInvalid("/health returned a non-JSON body") from error
        if payload.get("status") != "ok":
            self._health_state = "config_invalid"
            raise EmbeddingConfigInvalid(f"/health status is {payload.get('status')!r}, not 'ok'")
        if payload.get("dim") != self.dimension:
            self._health_state = "config_invalid"
            raise EmbeddingConfigInvalid(
                f"/health dim {payload.get('dim')!r} != configured dimension {self.dimension}"
            )
        service_model = normalize_model_basename(str(payload.get("model") or ""))
        if service_model != normalize_model_basename(self.model):
            self._health_state = "config_invalid"
            raise EmbeddingConfigInvalid(
                f"/health model basename {service_model!r} != configured {normalize_model_basename(self.model)!r}"
            )
        self._health_state = "ready"

    # ------------------------------------------------------------------
    # Encoding: identity (requirement 03 §7.2/§7.5) and context
    # (requirement 05 §7.2/§7.5), through one shared ladder below.
    # ------------------------------------------------------------------

    async def embed_identity(self, text: str, atom_type: str) -> list[float]:
        """BGE(pure literal) for one E/P literal; retries only transport/5xx."""
        if not text or not text.strip():
            raise EmbeddingError("empty_literal", "blank literal rejected before any HTTP call")
        endpoint = _IDENTITY_ENDPOINTS.get(atom_type)
        if endpoint is None:
            raise EmbeddingError(
                "unsupported_atom_type", f"identity vectors only exist for E/P, not {atom_type!r}"
            )
        return await self._post_and_validate(endpoint, {"term": text})

    async def embed_context(self, context_text: str) -> list[float]:
        """BGE(predicate-frame context) for anchor routing; never persisted."""
        if not context_text or not context_text.strip():
            raise EmbeddingError("empty_literal", "blank context text rejected before any HTTP call")
        return await self._post_and_validate(_CONTEXT_ENDPOINT, {"text": context_text})

    async def _post_and_validate(self, endpoint: str, body: dict[str, str]) -> list[float]:
        """One POST → parse → dimension/finite ladder, shared by both kinds.

        Retries only timeout/transport/5xx (up to ``max_retries``); 4xx/422
        and malformed payloads are never retried and never faked.
        """
        last_error: EmbeddingError | None = None
        for _attempt in range(1 + self.max_retries):
            try:
                response = await self._http.post(endpoint, json=body, headers=self._auth_headers)
            except httpx.TimeoutException as error:
                last_error = EmbeddingError("timeout", f"{endpoint} timed out: {type(error).__name__}")
                continue
            except httpx.TransportError as error:
                last_error = EmbeddingError("transport", f"{endpoint} transport error: {type(error).__name__}")
                continue
            if response.status_code >= 500:
                last_error = EmbeddingError("http_5xx", f"{endpoint} answered HTTP {response.status_code}")
                continue
            if response.status_code == 422:
                # Contract violation (requirement 03 §7.2): the frozen request
                # body can never trigger this — never retried.
                raise EmbeddingError(
                    "contract_violation", f"{endpoint} answered HTTP 422 for a frozen request shape"
                )
            if response.status_code >= 400:
                raise EmbeddingError("http_4xx", f"{endpoint} answered HTTP {response.status_code}")
            try:
                vector = response.json()["embedding"]
                values = [float(component) for component in vector]
            except Exception as error:
                raise EmbeddingError(
                    "bad_payload", f"{endpoint} returned an unparsable embedding payload"
                ) from error
            if len(values) != self.dimension:
                raise EmbeddingError(
                    "bad_dimension",
                    f"{endpoint} returned {len(values)} dimensions, expected {self.dimension}",
                )
            if not all(math.isfinite(component) for component in values):
                raise EmbeddingError(
                    "bad_payload", f"{endpoint} returned a non-finite embedding component"
                )
            return values
        assert last_error is not None  # retry loop always records before exhausting
        raise last_error
