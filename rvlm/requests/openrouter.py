import os

import openai

from rvlm import secrets
from rvlm.requests._media import build_openai_content
from rvlm.requests._pool import KeyPool

# OpenRouter key: prefer the env var (the documented setup), fall back to a
# secrets.py list for parity with the other providers.
_ENV_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_API_KEYS = ([_ENV_KEY] if _ENV_KEY else []) + list(getattr(secrets, "OPENROUTER_API_KEYS", []))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# thinking_level -> OpenRouter's normalized reasoning effort. OpenRouter maps
# this to each underlying provider's mechanism (OpenAI reasoning_effort,
# Anthropic thinking budget, ...). "OFF"/"NONE" disables reasoning.
_REASONING_EFFORT = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}


class OpenRouterResponse:
    """Wrapper providing .text to match the other provider interfaces."""

    def __init__(self, raw):
        self._raw = raw
        self.text = raw.choices[0].message.content


_key_pool = KeyPool(
    [openai.AsyncOpenAI(api_key=k, base_url=OPENROUTER_BASE_URL) for k in OPENROUTER_API_KEYS],
    name="OpenRouterKeyPool",
    empty_error=(
        "No OpenRouter API key configured. Set the OPENROUTER_API_KEY environment "
        "variable (or add OPENROUTER_API_KEYS to rvlm/secrets.py)."
    ),
)


def _resolve_slug(model_id: str) -> str:
    """Strip the routing prefix, leaving the OpenRouter model slug.

    Accepts "openrouter/<vendor>/<model>" or a bare "<vendor>/<model>".
    """
    return model_id[len("openrouter/"):] if model_id.startswith("openrouter/") else model_id


async def call_openrouter(
    prompt,
    video_input=None,
    img_input=None,
    thinking_level="MEDIUM",
    model_id: str = "openrouter/openai/gpt-5.6-terra",
    json_output: bool = False,
    response_schema=None,
    include_thoughts: bool = False,
):
    """
    Calls a model through OpenRouter's OpenAI-compatible endpoint. Works for any
    OpenRouter model slug (e.g. "openai/gpt-5.6-terra", "anthropic/claude-opus-4.8").

    Video is decomposed into per-frame JPEG images. JSON output is requested via
    a prompt instruction (provider-agnostic); markdown fences are stripped by
    the shared normalization in call_api.

    Args:
        prompt: String text
        video_input: Path to .mp4 file OR bytes data OR np.ndarray (T,H,W,C).
        img_input: Path to .jpg/.png file OR bytes data OR np.ndarray (H,W,C),
                   OR a list of any of the above.
        thinking_level: "OFF"/"NONE" (no reasoning), "LOW", "MEDIUM", or "HIGH"
                        — mapped to OpenRouter's normalized reasoning effort.
        model_id: "openrouter/<vendor>/<model>" or "<vendor>/<model>".
        json_output: If True, append a JSON-only instruction to the prompt.
        response_schema: Ignored (accepted for call_api parity).
        include_thoughts: Ignored (accepted for call_api parity).
    """
    text = prompt
    if json_output:
        text = f"{prompt}\n\nRespond with valid JSON only (a top-level array), with no markdown fences."

    kwargs = dict(
        model=_resolve_slug(model_id),
        messages=[{"role": "user", "content": build_openai_content(text, video_input, img_input)}],
    )

    effort = _REASONING_EFFORT.get((thinking_level or "MEDIUM").strip().upper())
    if effort is not None:
        # OpenRouter's normalized reasoning knob; translated per underlying provider.
        kwargs["extra_body"] = {"reasoning": {"effort": effort}}

    return await _key_pool.call(_create, **kwargs)


async def _create(client, **kwargs):
    return OpenRouterResponse(await client.chat.completions.create(**kwargs))
