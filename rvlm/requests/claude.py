from rvlm import secrets
from rvlm.requests._media import encode_b64, media_jpegs
from rvlm.requests._pool import KeyPool

ANTHROPIC_API_KEYS = getattr(secrets, "ANTHROPIC_API_KEYS", [])

# thinking_level -> extended-thinking token budget. Claude requires a budget of
# at least 1024 tokens when thinking is enabled; "OFF"/"NONE" disables it.
_THINKING_BUDGET = {
    "OFF": 0,
    "NONE": 0,
    "LOW": 2048,
    "MEDIUM": 4096,
    "HIGH": 8192,
}
_MAX_TOKENS = 8192


class ClaudeResponse:
    """Wrapper providing .text (and .thinking) to match the other providers."""

    def __init__(self, raw):
        self._raw = raw
        text_parts = []
        thinking_parts = []
        for block in getattr(raw, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype in ("thinking", "redacted_thinking"):
                thinking_parts.append(getattr(block, "thinking", ""))
        self.text = "".join(text_parts)
        self.thinking = "".join(thinking_parts)


def _budget_and_thinking(thinking_level: str):
    level = (thinking_level or "MEDIUM").strip().upper()
    budget = _THINKING_BUDGET.get(level, _THINKING_BUDGET["MEDIUM"])
    if budget <= 0:
        return None
    return {"type": "enabled", "budget_tokens": budget}


def _build_content(prompt, video_input, img_input) -> list:
    """Build Claude message content: images first, then the text prompt.

    Claude has no native video support, so video is decomposed into per-frame
    JPEG images. Claude's image block uses a base64 ``source`` (not a data URI).
    """
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encode_b64(jpeg)}}
        for jpeg in media_jpegs(video_input, img_input)
    ]
    content.append({"type": "text", "text": prompt})
    return content


# The pool is built lazily on first use so importing this module never requires
# the anthropic SDK unless a Claude model is actually called.
_key_pool = None


def _get_key_pool() -> KeyPool:
    global _key_pool
    if _key_pool is None:
        import anthropic

        _key_pool = KeyPool(
            [anthropic.AsyncAnthropic(api_key=k) for k in ANTHROPIC_API_KEYS],
            name="ClaudeKeyPool",
            empty_error=(
                "No Anthropic API keys configured. Add ANTHROPIC_API_KEYS to "
                "rvlm/secrets.py (or use an 'openrouter/anthropic/...' model)."
            ),
            fail_fast_on_400=False,
        )
    return _key_pool


async def _create(client, **kwargs):
    return ClaudeResponse(await client.messages.create(**kwargs))


async def call_claude(
    prompt,
    video_input=None,
    img_input=None,
    thinking_level="MEDIUM",
    model_id: str = "claude-sonnet-4-20250514",
    json_output: bool = False,
    response_schema=None,
    include_thoughts: bool = False,
):
    """
    Calls Claude via the direct Anthropic API with a video, one or more images,
    and a text prompt. Mirrors the call_gemini / call_gpt interface.

    Claude has no native video support, so video is decomposed into per-frame
    JPEG images (like the GPT provider). For quick debugging without Anthropic
    keys, an "openrouter/anthropic/..." model id routes the same request through
    OpenRouter instead.

    Args:
        prompt: String text
        video_input: Path to .mp4 file OR bytes data OR np.ndarray (T,H,W,C).
        img_input: Path to .jpg/.png file OR bytes data OR np.ndarray (H,W,C),
                   OR a list of any of the above.
        thinking_level: "OFF"/"NONE" (no thinking), "LOW", "MEDIUM", or "HIGH"
                        — mapped to an extended-thinking token budget.
        model_id: Anthropic model ID (e.g. "claude-sonnet-4-20250514").
        json_output: If True, force a top-level JSON array via assistant prefill
                     (or a prompt instruction when thinking is enabled).
        response_schema: Ignored (accepted for call_api parity).
        include_thoughts: Accepted for parity; thinking blocks are always exposed
                          via ClaudeResponse.thinking when thinking is enabled.
    """
    content = _build_content(prompt, video_input, img_input)
    messages = [{"role": "user", "content": content}]

    kwargs = dict(model=model_id, max_tokens=_MAX_TOKENS, messages=messages)

    thinking = _budget_and_thinking(thinking_level)
    if thinking is not None:
        # max_tokens must exceed the thinking budget.
        kwargs["max_tokens"] = _MAX_TOKENS + thinking["budget_tokens"]
        kwargs["thinking"] = thinking

    if json_output:
        # The Anthropic API has no JSON mode; assistant-message prefill of "["
        # forces a top-level JSON array. Prefill is incompatible with extended
        # thinking, so only apply it when thinking is disabled.
        if thinking is None:
            messages.append({"role": "assistant", "content": "["})
        else:
            content[-1]["text"] = f"{prompt}\n\nRespond with valid JSON only (a top-level array)."

    resp = await _get_key_pool().call(_create, **kwargs)

    if json_output and thinking is None and not resp.text.lstrip().startswith("["):
        # Reattach the prefilled "[" the API strips from the returned text.
        resp.text = "[" + resp.text
    return resp
