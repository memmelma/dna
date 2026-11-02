import httpx

from dna import secrets
from dna.requests._media import encode_b64, img_to_bytes, video_to_bytes
from dna.requests._pool import KeyPool

META_API_KEYS = getattr(secrets, "META_API_KEYS", [])

META_BASE_URL = "https://api.meta.ai/v1/responses"

_http = httpx.AsyncClient(timeout=120.0)


class MuseResponse:
    """Wrapper providing .text to match the other provider interfaces."""

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


# The "clients" are the raw API keys; _post uses each to authenticate one request.
_key_pool = KeyPool(
    list(META_API_KEYS),
    name="MuseKeyPool",
    empty_error="No Meta API keys configured. Add META_API_KEYS to dna/secrets.py.",
)


async def _post(api_key, *, payload):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = await _http.post(META_BASE_URL, headers=headers, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} {resp.reason_phrase}: {resp.text}")
    return MuseResponse(resp.json())


async def call_muse(
    prompt,
    video_input=None,
    img_input=None,
    thinking_level="MEDIUM",
    model_id: str = "muse-spark-1.1",
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
        json_output: If True, append a JSON-only instruction to the prompt
                     (Muse's documented API has no structured-output field).
        response_schema: Ignored (accepted for call_api parity).
    """
    text = prompt
    if json_output:
        text = f"{prompt}\n\nRespond with valid JSON only."
    content = [{"type": "input_text", "text": text}]

    if video_input is not None:
        content.append({
            "type": "input_video",
            "video_url": f"data:video/mp4;base64,{encode_b64(video_to_bytes(video_input))}",
        })

    if img_input is not None:
        img_inputs = img_input if isinstance(img_input, list) else [img_input]
        for img in img_inputs:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encode_b64(img_to_bytes(img))}",
            })

    payload = {
        "model": model_id,
        "input": [{"role": "user", "content": content}],
        "stream": False,
    }

    return await _key_pool.call(_post, payload=payload)
