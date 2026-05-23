import os
import io
import imageio
import asyncio
import numpy as np
from google import genai
from google.genai import types

from rvlm.secrets import GOOGLE_API_KEYS, GOOGLE_API_KEYS_PREVIEW

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
                # random delay to avoid hitting RPM limit
                await asyncio.sleep(np.random.uniform(1, 3))
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
_key_pool_preview = GeminiKeyPool(GOOGLE_API_KEYS_PREVIEW)

async def call_gemini(prompt, video_input=None, img_input=None, thinking_level="MEDIUM", model_id: str = "gemini-3-flash-preview", media_resolution="MEDIA_RESOLUTION_HIGH", json_output: bool = False, response_schema=None, include_thoughts=False):
    """
    Calls Gemini with a video, one or more images, and a text prompt.
    
    Args:
        video_input: Path to .mp4 file OR base64/bytes data
        img_input: Path to .jpg/.png file OR base64/bytes data OR np.ndarray,
                   OR a list of any of the above
        prompt: String text
        model_id: Gemini model ID
    """

    parts = []

    # (optional) process video
    if video_input is not None:
        if isinstance(video_input, np.ndarray):
            buf = io.BytesIO()
            imageio.mimwrite(buf, video_input, format="mp4", fps=1, macro_block_size=1)
            video_data = buf.getvalue()
        elif isinstance(video_input, str) and os.path.exists(video_input):
            with open(video_input, "rb") as f:
                video_data = f.read()
        else:
            video_data = video_input
        parts.append(types.Part.from_bytes(mime_type="video/mp4", data=video_data))

    # (optional) process image(s)
    if img_input is not None:
        img_inputs = img_input if isinstance(img_input, list) else [img_input]
        for img in img_inputs:
            if isinstance(img, np.ndarray):
                buf = io.BytesIO()
                imageio.imwrite(buf, img, format="JPEG", macro_block_size=1)
                img_data = buf.getvalue()
            elif isinstance(img, str) and os.path.exists(img):
                with open(img, "rb") as f:
                    img_data = f.read()
            else:
                img_data = img
            parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=img_data))

    # process prompt
    parts.append(types.Part.from_text(text=prompt))

    # configuration
    if "gemini-robotics-er-early-access" in model_id or "gemini-robotics-er-1.5-preview" in model_id or "gemini-robotics-er-1.6-preview" in model_id:
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level, include_thoughts=include_thoughts),
            media_resolution="MEDIA_RESOLUTION_HIGH",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            **({"response_mime_type": "application/json", 
                **({"response_schema": response_schema} if response_schema else {})} 
               if json_output else {}),
        )
        response = await _key_pool_preview.generate_content(
            model=model_id,
            contents=[types.Content(role="user", parts=parts)],
            config=generate_content_config,
        )
        return response
    else:
        if "gemini-3" in model_id:
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level, include_thoughts=include_thoughts),
                media_resolution="MEDIA_RESOLUTION_HIGH",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                **({"response_mime_type": "application/json", 
                    **({"response_schema": response_schema} if response_schema else {})} 
                if json_output else {}),
            )
        elif "gemini-2.5" in model_id:
            assert thinking_level == "HIGH", "Gemini 2.5 does not support thinking level < HIGH"
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1, include_thoughts=include_thoughts),
                media_resolution="MEDIA_RESOLUTION_HIGH",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                **({"response_mime_type": "application/json", 
                    **({"response_schema": response_schema} if response_schema else {})} 
                if json_output else {}),
            )
        else:
            raise ValueError(f"Unsupported model ID: {model_id}")

        response = await _key_pool.generate_content(
            model=model_id,
            contents=[types.Content(role="user", parts=parts)],
            config=generate_content_config,
        )
        return response
    