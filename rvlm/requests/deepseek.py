import asyncio

import numpy as np
import openai

from rvlm.secrets import DEEPSEEK_API_KEYS

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekResponse:
    """Wrapper providing .text to match Gemini/GPT response interface."""
    def __init__(self, raw):
        self._raw = raw
        self.text = raw.choices[0].message.content


class DeepSeekKeyPool:
    def __init__(self, api_keys: list, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
        self._clients = [openai.AsyncOpenAI(api_key=k, base_url=DEEPSEEK_BASE_URL) for k in api_keys]
        self._idx = 0
        self._lock = asyncio.Lock()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    async def _claim_next(self) -> tuple:
        """Atomically advance the index and return (idx, client) for this call."""
        async with self._lock:
            idx = self._idx
            self._idx = (self._idx + 1) % len(self._clients)
        return idx, self._clients[idx]

    async def _rotate(self, from_idx: int) -> None:
        """On quota exhaustion, skip past from_idx if no one else already did."""
        async with self._lock:
            if self._idx == (from_idx + 1) % len(self._clients):
                self._idx = (self._idx + 1) % len(self._clients)
                print(f"[DeepSeekKeyPool] quota hit on key {from_idx}, skipping to key {self._idx}")

    async def create(self, **kwargs):
        for attempt in range(self._max_retries + 1):
            idx, client = await self._claim_next()
            try:
                await asyncio.sleep(np.random.uniform(0.5, 1.5))
                resp = await client.chat.completions.create(**kwargs)
                return DeepSeekResponse(resp)
            except Exception as e:
                err = str(e)
                # 400s (e.g. malformed/unsupported content type) are deterministic;
                # retrying wastes 5 exponential-backoff rounds (~1 min) for nothing.
                if "Error code: 400" in err or "invalid_request_error" in err:
                    raise
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                print(f"[DeepSeekKeyPool] attempt {attempt + 1}/{self._max_retries + 1} failed (key {idx}): {err}, retrying in {delay:.1f}s")
                if attempt == self._max_retries:
                    raise
                if "429" in err or "rate_limit" in err.lower():
                    await self._rotate(idx)
                await asyncio.sleep(delay)


_key_pool = DeepSeekKeyPool(DEEPSEEK_API_KEYS)


async def call_deepseek(
    prompt,
    video_input=None,
    img_input=None,
    thinking_level="MEDIUM",
    model_id: str = "deepseek-v4-pro",
    media_resolution="low",
    json_output: bool = False,
    response_schema=None,
):
    """
    Calls DeepSeek with a text prompt.

    NOTE: DeepSeek's public API (api.deepseek.com) is TEXT-ONLY — confirmed via
    debug_new_apis.py, which got a hard 400 ("unknown variant `image_url`,
    expected `text`") for both image and video inputs. There is no vision
    support at all (not even frames-as-images like call_gpt). video_input /
    img_input are therefore rejected up front with a clear error instead of
    silently failing against the API repeatedly.

    Args:
        prompt: String text
        video_input: NOT SUPPORTED (DeepSeek has no vision endpoint). Passing
            this raises ValueError.
        img_input: NOT SUPPORTED (DeepSeek has no vision endpoint). Passing
            this raises ValueError.
        thinking_level: "LOW", "MEDIUM", or "HIGH" — mapped to reasoning_effort.
        model_id: DeepSeek model ID (e.g. "deepseek-v4-pro", "deepseek-chat")
        media_resolution: Ignored (Gemini/GPT-only); accepted for call_api parity.
        json_output: If True, request JSON output format.
        response_schema: Optional JSON schema dict for structured output.
    """
    if video_input is not None or img_input is not None:
        raise ValueError(
            "DeepSeek's public API is text-only (no vision support for images or "
            "video); call_deepseek cannot be used for video/image-grounded modalities. "
            "Use a text-only modality (e.g. progress-from-description) with DeepSeek instead."
        )

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ]

    # build request kwargs
    kwargs = dict(
        model=model_id,
        messages=messages,
        stream=False,
        reasoning_effort=thinking_level.lower(),
        extra_body={"thinking": {"type": "enabled"}},
    )

    # JSON / structured output
    if json_output:
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

    return await _key_pool.create(**kwargs)
