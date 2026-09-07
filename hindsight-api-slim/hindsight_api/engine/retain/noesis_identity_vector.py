"""Noesis identity-vector client (requirement 03).

Write-only bge identity vectors: ``BGE(pure literal)`` — the atom text itself,
no context, no prefixes, no role/type concatenation. Endpoint mapping is
frozen (requirement 03 §7.3): ``E`` → ``/normalize/entity``, ``P`` →
``/normalize/predicate``. ``/normalize/token`` is never used (it returns a
contextual hidden-state pooling, not an identity vector) and
``/normalize/sentence`` is reserved for requirement 05.

Failure semantics (requirement 03 §7.4/§7.5/§9): every literal is encoded on
its own, transport errors and 5xx are retried up to ``max_retries``, 4xx/422
are never retried, a wrong response dimension is a per-literal failure, and
there is no fallback — no zero vectors, no random vectors, no local model.
The first use per process runs ``GET /health``; config-level mismatches
(dim/model/status) disable vectors for the process (cached), transport
failures are not cached and the next batch retries.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Frozen endpoint map (requirement 03 §7.3). G never gets a vector.
_IDENTITY_ENDPOINTS = {"E": "/normalize/entity", "P": "/normalize/predicate"}

# error_kind values recorded in identity_vector alert details.
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


class IdentityConfigInvalid(Exception):
    """The service /health contradicts the configured generation.

    Process-level disable (cached on the client instance); facts keep flowing
    with NULL embeddings and an ``identity_vector_config_invalid`` alert.
    """


class IdentityServiceUnavailable(Exception):
    """Transport-level /health failure (never cached; next batch retries)."""


class IdentityEmbedError(Exception):
    """A single literal could not be embedded; it registers with NULL vector."""

    def __init__(self, error_kind: str, message: str) -> None:
        super().__init__(message)
        self.error_kind = error_kind


class NoesisIdentityClient:
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
            raise IdentityConfigInvalid("identity service /health failed config validation (cached)")
        try:
            response = await self._http.get("/health", headers=self._auth_headers)
        except httpx.TransportError as error:
            raise IdentityServiceUnavailable(f"/health transport failure: {type(error).__name__}") from error
        if response.status_code != 200:
            # A sick service may recover; do not cache, do not guess.
            raise IdentityServiceUnavailable(f"/health answered HTTP {response.status_code}")
        try:
            payload = response.json()
        except Exception as error:
            self._health_state = "config_invalid"
            raise IdentityConfigInvalid("/health returned a non-JSON body") from error
        if payload.get("status") != "ok":
            self._health_state = "config_invalid"
            raise IdentityConfigInvalid(f"/health status is {payload.get('status')!r}, not 'ok'")
        if payload.get("dim") != self.dimension:
            self._health_state = "config_invalid"
            raise IdentityConfigInvalid(
                f"/health dim {payload.get('dim')!r} != configured dimension {self.dimension}"
            )
        service_model = normalize_model_basename(str(payload.get("model") or ""))
        if service_model != normalize_model_basename(self.model):
            self._health_state = "config_invalid"
            raise IdentityConfigInvalid(
                f"/health model basename {service_model!r} != configured {normalize_model_basename(self.model)!r}"
            )
        self._health_state = "ready"

    # ------------------------------------------------------------------
    # Identity encoding (requirement 03 §7.2/§7.5)
    # ------------------------------------------------------------------

    async def embed(self, text: str, atom_type: str) -> list[float]:
        """BGE(pure literal) for one E/P literal; retries only transport/5xx."""
        if not text or not text.strip():
            raise IdentityEmbedError("empty_literal", "blank literal rejected before any HTTP call")
        endpoint = _IDENTITY_ENDPOINTS.get(atom_type)
        if endpoint is None:
            raise IdentityEmbedError(
                "unsupported_atom_type", f"identity vectors only exist for E/P, not {atom_type!r}"
            )
        last_error: IdentityEmbedError | None = None
        for _attempt in range(1 + self.max_retries):
            try:
                response = await self._http.post(endpoint, json={"term": text}, headers=self._auth_headers)
            except httpx.TimeoutException as error:
                last_error = IdentityEmbedError("timeout", f"{endpoint} timed out: {type(error).__name__}")
                continue
            except httpx.TransportError as error:
                last_error = IdentityEmbedError("transport", f"{endpoint} transport error: {type(error).__name__}")
                continue
            if response.status_code >= 500:
                last_error = IdentityEmbedError("http_5xx", f"{endpoint} answered HTTP {response.status_code}")
                continue
            if response.status_code == 422:
                # Contract violation (requirement 03 §7.2): the frozen request
                # body can never trigger this — never retried.
                raise IdentityEmbedError(
                    "contract_violation", f"{endpoint} answered HTTP 422 for a frozen request shape"
                )
            if response.status_code >= 400:
                raise IdentityEmbedError("http_4xx", f"{endpoint} answered HTTP {response.status_code}")
            try:
                vector = response.json()["embedding"]
                values = [float(component) for component in vector]
            except Exception as error:
                raise IdentityEmbedError(
                    "bad_payload", f"{endpoint} returned an unparsable embedding payload"
                ) from error
            if len(values) != self.dimension:
                raise IdentityEmbedError(
                    "bad_dimension",
                    f"{endpoint} returned {len(values)} dimensions, expected {self.dimension}",
                )
            if not all(math.isfinite(component) for component in values):
                raise IdentityEmbedError(
                    "bad_payload", f"{endpoint} returned a non-finite embedding component"
                )
            return values
        assert last_error is not None  # retry loop always records before exhausting
        raise last_error
