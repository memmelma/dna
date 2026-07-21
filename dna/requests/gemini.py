from google import genai
from google.genai import types

from dna import secrets
from dna.requests._media import img_to_bytes, video_to_bytes
from dna.requests._pool import KeyPool

GOOGLE_API_KEYS = getattr(secrets, "GOOGLE_API_KEYS", [])
GOOGLE_API_KEYS_PREVIEW = getattr(secrets, "GOOGLE_API_KEYS_PREVIEW", [])

# Robotics-ER preview models use a separate key pool (early-access quota).
_PREVIEW_MODELS = (
    "gemini-robotics-er-early-access",
    "gemini-robotics-er-1.5-preview",
    "gemini-robotics-er-1.6-preview",
)

# Pools are built lazily on first use so importing this module never spins up
# genai.Client objects for keys that are never used.
_pools: dict = {}


def _get_key_pool(preview: bool) -> KeyPool:
    key = "preview" if preview else "standard"
    if key not in _pools:
        api_keys = GOOGLE_API_KEYS_PREVIEW if preview else GOOGLE_API_KEYS
        _pools[key] = KeyPool(
            [genai.Client(api_key=k) for k in api_keys],
            name="GeminiKeyPool",
            empty_error="No Google API keys configured. Add GOOGLE_API_KEYS to dna/secrets.py.",
            jitter=(1, 3),
            fail_fast_on_400=False,
        )
    return _pools[key]


def _make_config(model_id, thinking_level, json_output, response_schema, include_thoughts):
    """Build the GenerateContentConfig for a given Gemini model family."""
    if "gemini-2.5" in model_id and not any(m in model_id for m in _PREVIEW_MODELS):
        if thinking_level != "HIGH":
            raise ValueError("Gemini 2.5 does not support thinking level < HIGH")
        thinking_config = types.ThinkingConfig(thinking_budget=-1, include_thoughts=include_thoughts)
    else:
        thinking_config = types.ThinkingConfig(thinking_level=thinking_level, include_thoughts=include_thoughts)

    json_cfg = {}
    if json_output:
        json_cfg = {"response_mime_type": "application/json"}
        if response_schema:
            json_cfg["response_schema"] = response_schema

    return types.GenerateContentConfig(
        thinking_config=thinking_config,
        media_resolution="MEDIA_RESOLUTION_HIGH",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        **json_cfg,
    )


async def _generate(client, **kwargs):
    return await client.aio.models.generate_content(**kwargs)


async def call_gemini(prompt, video_input=None, img_input=None, thinking_level="MEDIUM", model_id: str = "gemini-3-flash-preview", json_output: bool = False, response_schema=None, include_thoughts=False):
    """
    Calls Gemini with a video, one or more images, and a text prompt.

    Gemini has native video support and always uses MEDIA_RESOLUTION_HIGH, so
    there is no media-resolution knob.

    Args:
        video_input: Path to .mp4 file OR base64/bytes data OR np.ndarray
        img_input: Path to .jpg/.png file OR base64/bytes data OR np.ndarray,
                   OR a list of any of the above
        prompt: String text
        model_id: Gemini model ID
    """
    parts = []
    if video_input is not None:
        parts.append(types.Part.from_bytes(mime_type="video/mp4", data=video_to_bytes(video_input)))
    if img_input is not None:
        img_inputs = img_input if isinstance(img_input, list) else [img_input]
        for img in img_inputs:
            parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=img_to_bytes(img)))
    parts.append(types.Part.from_text(text=prompt))

    is_preview = any(m in model_id for m in _PREVIEW_MODELS)
    if not is_preview and "gemini-3" not in model_id and "gemini-2.5" not in model_id:
        raise ValueError(f"Unsupported model ID: {model_id}")

    config = _make_config(model_id, thinking_level, json_output, response_schema, include_thoughts)
    return await _get_key_pool(is_preview).call(
        _generate,
        model=model_id,
        contents=[types.Content(role="user", parts=parts)],
        config=config,
    )
