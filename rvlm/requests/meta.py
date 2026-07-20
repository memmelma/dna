import asyncio
import base64
import io
import os

import httpx
import imageio
import numpy as np

from rvlm.secrets import META_API_KEYS

META_BASE_URL = "https://api.meta.ai/v1/responses"


class MuseResponse:
    """Wrapper providing .text to match Gemini/GPT response interface."""
    def __init__(self, raw: dict):
        self._raw = raw
        self.text = _extract_text(raw)


def _extract_text(raw: dict) -> str:
    """Extract the assistant's text from a Responses-API-style payload.

    Mirrors the shape used by OpenAI's Responses API (``output`` is a list of
    items, each with a ``content`` list of typed parts). Falls back to a few
    other plausible shapes in case Muse's schema differs slightly.
    """
    if "output_text" in raw and raw["output_text"]:
        return raw["output_text"]

    output = raw.get("output") or raw.get("response", {}).get("output")
    if output:
        for item in output:
            for part in item.get("content", []):
                if part.get("type") in ("output_text", "text") and part.get("text"):
                    return part["text"]

    # Last-resort fallbacks seen in other chat-style APIs.
    if "text" in raw:
        return raw["text"]
    choices = raw.get("choices")
    if choices:
        return choices[0].get("message", {}).get("content", "")

    raise ValueError(f"Could not extract text from Muse response: {raw!r}")


class MuseKeyPool:
    def __init__(self, api_keys: list, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
        self._api_keys = api_keys
        self._idx = 0
        self._lock = asyncio.Lock()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._client = httpx.AsyncClient(timeout=120.0)

    async def _claim_next(self) -> tuple:
        """Atomically advance the index and return (idx, api_key) for this call."""
        async with self._lock:
            idx = self._idx
            self._idx = (self._idx + 1) % len(self._api_keys)
        return idx, self._api_keys[idx]

    async def _rotate(self, from_idx: int) -> None:
        """On quota exhaustion, skip past from_idx if no one else already did."""
        async with self._lock:
            if self._idx == (from_idx + 1) % len(self._api_keys):
                self._idx = (self._idx + 1) % len(self._api_keys)
                print(f"[MuseKeyPool] quota hit on key {from_idx}, skipping to key {self._idx}")

    async def create(self, payload: dict):
        for attempt in range(self._max_retries + 1):
            idx, api_key = await self._claim_next()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            try:
                await asyncio.sleep(np.random.uniform(0.5, 1.5))
                resp = await self._client.post(META_BASE_URL, headers=headers, json=payload)
                if resp.status_code >= 400:
                    raise RuntimeError(f"{resp.status_code} {resp.reason_phrase}: {resp.text}")
                return MuseResponse(resp.json())
            except Exception as e:
                err = str(e)
                # 400s (e.g. malformed payload) are deterministic; retrying wastes
                # 5 exponential-backoff rounds (~1 min) for nothing.
                if "400 Bad Request" in err or "invalid_request_error" in err:
                    raise
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                print(f"[MuseKeyPool] attempt {attempt + 1}/{self._max_retries + 1} failed (key {idx}): {err}, retrying in {delay:.1f}s")
                if attempt == self._max_retries:
                    raise
                if "429" in err or "rate_limit" in err.lower():
                    await self._rotate(idx)
                await asyncio.sleep(delay)


_key_pool = MuseKeyPool(META_API_KEYS)


def _encode_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _video_to_bytes(video_input) -> bytes:
    if isinstance(video_input, np.ndarray):
        buf = io.BytesIO()
        imageio.mimwrite(buf, video_input, format="mp4", fps=1, macro_block_size=1)
        return buf.getvalue()
    if isinstance(video_input, str) and os.path.exists(video_input):
        with open(video_input, "rb") as f:
            return f.read()
    return video_input if isinstance(video_input, bytes) else bytes(video_input)


def _img_to_bytes(img) -> bytes:
    if isinstance(img, np.ndarray):
        buf = io.BytesIO()
        imageio.imwrite(buf, img, format="JPEG")
        return buf.getvalue()
    if isinstance(img, str) and os.path.exists(img):
        with open(img, "rb") as f:
            return f.read()
    return img if isinstance(img, bytes) else bytes(img)


async def call_muse(
    prompt,
    video_input=None,
    img_input=None,
    thinking_level="MEDIUM",
    model_id: str = "muse-spark-1.1",
    media_resolution="low",
    json_output: bool = False,
    response_schema=None,
):
    """
    Calls Meta Muse Spark via the /v1/responses endpoint with a video, one or
    more images, and a text prompt. Muse supports native video input, so
    video_input is sent as a single input_video part (not decomposed into frames).

    Args:
        prompt: String text
        video_input: Path to .mp4 file OR bytes data OR np.ndarray (T,H,W,C).
                      Sent natively as base64 video (no frame decomposition).
        img_input: Path to .jpg/.png file OR bytes data OR np.ndarray (H,W,C),
                   OR a list of any of the above.
        thinking_level: Accepted for call_api parity; Muse's public API does not
                         document a reasoning-effort knob, so it is currently unused.
        model_id: Muse model ID (e.g. "muse-spark-1.1")
        media_resolution: Ignored (Gemini-only); accepted for call_api parity.
        json_output: If True, append a JSON-only instruction to the prompt
                     (Muse's documented API has no structured-output field).
        response_schema: Ignored (accepted for call_api parity).
    """
    _ = (thinking_level, media_resolution, response_schema)
    content = []

    text = prompt
    if json_output:
        text = f"{prompt}\n\nRespond with valid JSON only."
    content.append({"type": "input_text", "text": text})

    if video_input is not None:
        video_bytes = _video_to_bytes(video_input)
        content.append({
            "type": "input_video",
            "video_url": f"data:video/mp4;base64,{_encode_to_b64(video_bytes)}",
        })

    if img_input is not None:
        img_inputs = img_input if isinstance(img_input, list) else [img_input]
        for img in img_inputs:
            img_bytes = _img_to_bytes(img)
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{_encode_to_b64(img_bytes)}",
            })

    payload = {
        "model": model_id,
        "input": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "stream": False,
    }

    return await _key_pool.create(payload)
