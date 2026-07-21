"""Generic round-robin key pool with exponential-backoff retries.

Every provider used to carry its own near-identical ``*KeyPool`` class. They
differed only in the client type, the API call, the retry jitter, and which
error strings count as rate limits. ``KeyPool`` captures the shared machinery;
each provider supplies the per-call function and any overrides.
"""

import asyncio

import numpy as np

# Substrings that indicate a retryable rate-limit / quota error across providers.
_RATE_LIMIT_MARKERS = ("429", "rate_limit", "resource_exhausted", "overloaded")
# Substrings that indicate a deterministic 400 (retrying is pointless).
_BAD_REQUEST_MARKERS = ("error code: 400", "400 bad request", "invalid_request_error")


class KeyPool:
    """Round-robin over a list of clients with backoff, rotation, and jitter.

    Args:
        clients: Per-key client objects (or raw keys for providers that build
            requests by hand, e.g. Muse). Empty is allowed; the first call then
            raises ``empty_error`` so import never fails on missing keys.
        name: Label used in retry log lines.
        empty_error: Message raised when a call is made with no clients.
        jitter: (low, high) seconds of random pre-request sleep to smooth RPM.
        fail_fast_on_400: Raise immediately on deterministic 400s instead of
            burning the retry budget.
    """

    def __init__(
        self,
        clients: list,
        *,
        name: str = "KeyPool",
        empty_error: str = "No API keys configured.",
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: tuple = (0.5, 1.5),
        fail_fast_on_400: bool = True,
    ):
        self._clients = list(clients)
        self._name = name
        self._empty_error = empty_error
        self._idx = 0
        self._lock = asyncio.Lock()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter
        self._fail_fast_on_400 = fail_fast_on_400

    async def _claim_next(self) -> tuple:
        """Atomically advance the index and return (idx, client) for this call."""
        if not self._clients:
            raise RuntimeError(self._empty_error)
        async with self._lock:
            idx = self._idx
            self._idx = (self._idx + 1) % len(self._clients)
        return idx, self._clients[idx]

    async def _rotate(self, from_idx: int) -> None:
        """On quota exhaustion, skip past from_idx if no one else already did."""
        async with self._lock:
            if self._idx == (from_idx + 1) % len(self._clients):
                self._idx = (self._idx + 1) % len(self._clients)
                print(f"[{self._name}] quota hit on key {from_idx}, skipping to key {self._idx}")

    async def call(self, fn, **kwargs):
        """Run ``await fn(client, **kwargs)`` with retry / rotation / backoff.

        ``fn`` receives the claimed client and should perform the request and
        return the (already wrapped) response.
        """
        for attempt in range(self._max_retries + 1):
            idx, client = await self._claim_next()
            try:
                await asyncio.sleep(np.random.uniform(*self._jitter))
                return await fn(client, **kwargs)
            except Exception as e:
                err = str(e)
                if self._fail_fast_on_400 and any(m in err.lower() for m in _BAD_REQUEST_MARKERS):
                    raise
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                print(f"[{self._name}] attempt {attempt + 1}/{self._max_retries + 1} failed (key {idx}): {err}, retrying in {delay:.1f}s")
                if attempt == self._max_retries:
                    raise
                if any(m in err.lower() for m in _RATE_LIMIT_MARKERS):
                    await self._rotate(idx)
                await asyncio.sleep(delay)
