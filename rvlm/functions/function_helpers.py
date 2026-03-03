import os
import io
import re
import json
import asyncio
import imageio
import numpy as np
from google import genai
from google.genai import types

GOOGLE_API_KEYS = [
    # gemini-er-3
    # "AIzaSyCa83aP2XNC2cCZufGchURsY0zaJhRqx0g",
    # gemini-er-2
    "AIzaSyCr4aa7RsUW1zRCcJACVS6M6J3A08Uy1M0",
    # gemini-er-0
    "AIzaSyAIyKWeVkmuRLxIjuuLJglrts0TVv4eSco",
    # gemini-er-1
    "AIzaSyCu84vRyQrfMnCkVqxpOydodfEfobPjWO0",
    # gemini-er
    "AIzaSyB5Zm04vgtNo0C3-dHM6BTuLPLj--fYLl4",
]

class GeminiKeyPool:
    def __init__(self, api_keys: list, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
        self._clients = [genai.Client(api_key=k) for k in api_keys]
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
                print(f"[GeminiKeyPool] quota hit on key {from_idx}, skipping to key {self._idx}")

    async def generate_content(self, **kwargs):
        for attempt in range(self._max_retries + 1):
            idx, client = await self._claim_next()
            try:
                return await client.aio.models.generate_content(**kwargs)
            except Exception as e:
                err = str(e)
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                print(f"[GeminiKeyPool] attempt {attempt + 1}/{self._max_retries + 1} failed (key {idx}): {err}, retrying in {delay:.1f}s")
                if attempt == self._max_retries:
                    raise
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    await self._rotate(idx)
                await asyncio.sleep(delay)

_key_pool = GeminiKeyPool(GOOGLE_API_KEYS)

from rvlm.annotator.annotator_reward import SimpleCoTrackerDTW
_tracker: SimpleCoTrackerDTW = None

def _get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = SimpleCoTrackerDTW()
    return _tracker

def _parse_json(text):
    if isinstance(text, list):
        return json.dumps(text)

    if not text or not text.strip():
        raise ValueError("Cannot parse empty Gemini response as JSON")

    # 1. Try to extract from ```json ... ``` code fence
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                print(f"[_parse_json] code fence content is invalid JSON ({e}), trying full text")

    # 2. Fall back to parsing the full text directly
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse Gemini response as JSON: {e}\nRaw text:\n{repr(text)}")

async def call_gemini(prompt, video_input=None, img_input=None, thinking_level="MEDIUM"):
    """
    Calls Gemini with a video, an image, and a text prompt.
    
    Args:
        video_input: Path to .mp4 file OR base64/bytes data
        img_input: Path to .jpg/.png file OR base64/bytes data
        prompt: String text
    """
    model_id = "gemini-3-flash-preview"

    parts = []

    # (optional) process video
    if video_input is not None:
        if isinstance(video_input, np.ndarray):
            buf = io.BytesIO()
            imageio.mimwrite(buf, video_input, format="mp4", fps=1)
            video_data = buf.getvalue()
        elif isinstance(video_input, str) and os.path.exists(video_input):
            with open(video_input, "rb") as f:
                video_data = f.read()
        else:
            video_data = video_input
        parts.append(types.Part.from_bytes(mime_type="video/mp4", data=video_data))

    # (optional) process image
    if img_input is not None:
        if isinstance(img_input, np.ndarray):
            buf = io.BytesIO()
            imageio.imwrite(buf, img_input, format="JPEG")
            img_data = buf.getvalue()
        elif isinstance(img_input, str) and os.path.exists(img_input):
            with open(img_input, "rb") as f:
                img_data = f.read()
        else:
            img_data = img_input
        parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=img_data))

    # process prompt
    parts.append(types.Part.from_text(text=prompt))

    # configuration
    if model_id == "gemini-3-flash-preview":
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level, include_thoughts=True),
            # thinking_config=types.ThinkingConfig(thinking_level="HIGH", include_thoughts=True),
            media_resolution="MEDIA_RESOLUTION_HIGH",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
    # elif model_id == "gemini-2.5-flash" or model_id == "gemini-2.5-flash-lite":
    #     generate_content_config = types.GenerateContentConfig(
    #         media_resolution="MEDIA_RESOLUTION_HIGH",
    #         automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    #     )
    else:
        raise ValueError(f"Unsupported model ID: {model_id}")
    response = await _key_pool.generate_content(
        model=model_id,
        contents=[types.Content(role="user", parts=parts)],
        config=generate_content_config,
    )
    return response