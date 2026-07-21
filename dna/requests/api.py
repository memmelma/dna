import json
import logging
import re

logging.getLogger("httpx").setLevel(logging.WARNING)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence if the model wrapped its
    JSON in ```json ... ```. Leaves already-bare JSON untouched."""
    if not text:
        return text
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_RE.sub("", stripped)
    return stripped


def _normalize_json_array(response):
    """Normalize a JSON-output response's ``.text`` for uniform downstream parsing.

    dna's prompts ask for a top-level JSON array (callers do
    ``json.loads(res.text)[0]``), but models may (a) wrap the JSON in markdown
    code fences, or (b) return a bare object ``{...}`` instead of ``[{...}]``.
    Strip fences then wrap bare objects so every provider behaves identically.

    Only writes ``.text`` back when the value actually changed — some provider
    responses (e.g. Gemini's native object) expose ``.text`` as a read-only
    property, and a well-formed array response needs no rewrite anyway.
    """
    original = getattr(response, "text", None)
    if original is None:
        return response
    text = _strip_fences(original)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        text = original  # not JSON we recognize; leave untouched
    else:
        if isinstance(parsed, dict):
            text = json.dumps([parsed])
    if text != original:
        response.text = text
    return response


async def call_api(
    prompt,
    video_input=None,
    img_input=None,
    thinking_level="MEDIUM",
    model_id: str = "gemini-3-flash-preview",
    json_output: bool = False,
    response_schema=None,
    include_thoughts: bool = False,
):
    """Unified dispatcher across all VLM providers.

    Routing is by ``model_id`` prefix. Every provider returns an object with a
    ``.text`` attribute holding the model's textual output. Each provider is
    responsible for its own media-resolution strategy (Gemini uses HIGH, the
    frame-decomposing providers use low), so there is no ``media_resolution``
    knob here.

    Model ID prefixes:
        gemini*          -> Google Gemini (native video)
        openrouter/*     -> OpenRouter OpenAI-compatible endpoint (frames-as-images);
                            handy for debugging GPT / Claude through one endpoint
        gpt* / o*        -> OpenAI (frames-as-images)
        claude*          -> Anthropic API (frames-as-images)
        muse*            -> Meta Muse Spark (native video)
        qwen* / Qwen/*   -> local Qwen-VL via HuggingFace
    """
    model_lower = model_id.lower()

    if model_id.startswith("openrouter/"):
        from dna.requests.openrouter import call_openrouter

        response = await call_openrouter(
            prompt,
            video_input,
            img_input,
            thinking_level,
            model_id,
            json_output=json_output,
            response_schema=response_schema,
            include_thoughts=include_thoughts,
        )
    elif model_id.startswith("gemini"):
        from dna.requests.gemini import call_gemini

        response = await call_gemini(
            prompt,
            video_input,
            img_input,
            thinking_level,
            model_id,
            json_output=json_output,
            response_schema=response_schema,
            include_thoughts=include_thoughts,
        )
    elif model_id.startswith("claude"):
        from dna.requests.claude import call_claude

        response = await call_claude(
            prompt,
            video_input,
            img_input,
            thinking_level,
            model_id,
            json_output=json_output,
            response_schema=response_schema,
            include_thoughts=include_thoughts,
        )
    elif model_id.startswith("gpt") or model_id.startswith(("o1", "o3", "o4")):
        from dna.requests.gpt import call_gpt

        response = await call_gpt(
            prompt,
            video_input,
            img_input,
            thinking_level,
            model_id,
            json_output=json_output,
            response_schema=response_schema,
        )
    elif model_id.startswith("muse"):
        from dna.requests.meta import call_muse

        response = await call_muse(
            prompt,
            video_input,
            img_input,
            thinking_level,
            model_id,
            json_output=json_output,
            response_schema=response_schema,
        )
    elif model_lower.startswith("qwen") or model_id.startswith("Qwen/"):
        from dna.requests.qwen_hf import call_qwen_hf, _resolve_hf_model_id

        hf_id_lower = _resolve_hf_model_id(model_id).lower()
        if "thinking" in hf_id_lower:
            # Separate *-Thinking checkpoint (e.g. Qwen3-VL-8B-Thinking): always thinks.
            enable_thinking = True
        elif "instruct" in hf_id_lower:
            # Separate *-Instruct checkpoint: never thinks regardless of thinking_level.
            enable_thinking = False
        else:
            # Unified checkpoint: thinking is toggled via the chat template,
            # driven by thinking_level like Gemini/GPT.
            enable_thinking = (thinking_level or "MEDIUM").strip().upper() not in ("OFF", "NONE", "FALSE", "NO")
        response = await call_qwen_hf(
            prompt,
            video_input=video_input,
            img_input=img_input,
            thinking_level=thinking_level,
            model_id=model_id,
            json_output=json_output,
            enable_thinking=enable_thinking,
        )
    else:
        raise ValueError(f"Invalid model ID: {model_id}")

    if json_output:
        response = _normalize_json_array(response)
    return response
