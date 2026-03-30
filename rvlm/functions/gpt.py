import asyncio
import base64
import io
import os

import imageio
import numpy as np
import openai


OPENAI_API_KEYS = [
]


class GPTResponse:
    """Wrapper providing .text to match Gemini response interface."""
    def __init__(self, raw):
        self._raw = raw
        self.text = raw.choices[0].message.content


class GPTKeyPool:
    def __init__(self, api_keys: list, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
        self._clients = [openai.AsyncOpenAI(api_key=k) for k in api_keys]
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
                print(f"[GPTKeyPool] quota hit on key {from_idx}, skipping to key {self._idx}")

    async def create(self, **kwargs):
        for attempt in range(self._max_retries + 1):
            idx, client = await self._claim_next()
            try:
                await asyncio.sleep(np.random.uniform(0.5, 1.5))
                resp = await client.chat.completions.create(**kwargs)
                return GPTResponse(resp)
            except Exception as e:
                err = str(e)
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                print(f"[GPTKeyPool] attempt {attempt + 1}/{self._max_retries + 1} failed (key {idx}): {err}, retrying in {delay:.1f}s")
                if attempt == self._max_retries:
                    raise
                if "429" in err or "rate_limit" in err.lower():
                    await self._rotate(idx)
                await asyncio.sleep(delay)


_key_pool = GPTKeyPool(OPENAI_API_KEYS)


def _encode_image_to_b64(img_data: bytes) -> str:
    return base64.b64encode(img_data).decode("utf-8")


def _make_image_part(b64: str, detail: str = "high") -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{b64}",
            "detail": detail,
        },
    }


async def call_gpt(
    prompt,
    video_input=None,
    img_input=None,
    thinking_level="MEDIUM",
    model_id: str = "gpt-4o",
    json_output: bool = False,
    response_schema=None,
):
    """
    Calls GPT with a video, one or more images, and a text prompt.
    Mirrors the call_gemini interface.

    Args:
        prompt: String text
        video_input: Path to .mp4 file OR bytes data OR np.ndarray (T,H,W,C).
                     Frames are extracted and sent as individual images
                     (GPT has no native video support).
        img_input: Path to .jpg/.png file OR bytes data OR np.ndarray (H,W,C),
                   OR a list of any of the above.
        thinking_level: "LOW", "MEDIUM", or "HIGH" — mapped to reasoning_effort
                        for o-series models; ignored for gpt-4o.
        model_id: OpenAI model ID (e.g. "gpt-4o", "o3", "o4-mini")
        json_output: If True, request JSON output format.
        response_schema: Optional JSON schema dict for structured output.
    """
    content = []

    # (optional) video → individual frame images
    if video_input is not None:
        if isinstance(video_input, np.ndarray):
            frames = video_input
        elif isinstance(video_input, str) and os.path.exists(video_input):
            frames = np.stack(imageio.mimread(video_input, memtest=False))
        else:
            buf = io.BytesIO(video_input if isinstance(video_input, bytes) else video_input)
            frames = np.stack(imageio.mimread(buf, format="mp4", memtest=False))

        for frame in frames:
            buf = io.BytesIO()
            imageio.imwrite(buf, frame, format="JPEG")
            content.append(_make_image_part(_encode_image_to_b64(buf.getvalue())))

    # (optional) image(s)
    if img_input is not None:
        img_inputs = img_input if isinstance(img_input, list) else [img_input]
        for img in img_inputs:
            if isinstance(img, np.ndarray):
                buf = io.BytesIO()
                imageio.imwrite(buf, img, format="JPEG")
                img_data = buf.getvalue()
            elif isinstance(img, str) and os.path.exists(img):
                with open(img, "rb") as f:
                    img_data = f.read()
            else:
                img_data = img
            content.append(_make_image_part(_encode_image_to_b64(img_data)))

    # text prompt
    content.append({"type": "text", "text": prompt})

    # build request kwargs
    kwargs = dict(
        model=model_id,
        messages=[{"role": "user", "content": content}],
    )

    # reasoning_effort for o-series models (o1, o3, o4-mini, etc.)
    if model_id.startswith(("o1", "o3", "o4")):
        kwargs["reasoning_effort"] = thinking_level.lower()

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