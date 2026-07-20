import openai

from rvlm import secrets
from rvlm.requests._media import build_openai_content
from rvlm.requests._pool import KeyPool

OPENAI_API_KEYS = getattr(secrets, "OPENAI_API_KEYS", [])


class GPTResponse:
    """Wrapper providing .text to match the other provider interfaces."""

    def __init__(self, raw):
        self._raw = raw
        self.text = raw.choices[0].message.content


_key_pool = KeyPool(
    [openai.AsyncOpenAI(api_key=k) for k in OPENAI_API_KEYS],
    name="GPTKeyPool",
    empty_error="No OpenAI API keys configured. Add OPENAI_API_KEYS to rvlm/secrets.py.",
)


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

    GPT has no native video support, so video is decomposed into per-frame
    images sent at ``detail="low"`` (the resolution rvlm uses everywhere).

    Args:
        prompt: String text
        video_input: Path to .mp4 file OR bytes data OR np.ndarray (T,H,W,C).
                     Frames are extracted and sent as individual images.
        img_input: Path to .jpg/.png file OR bytes data OR np.ndarray (H,W,C),
                   OR a list of any of the above.
        thinking_level: "LOW", "MEDIUM", or "HIGH" — mapped to reasoning_effort
                        for reasoning models (o-series, gpt-5); ignored otherwise.
        model_id: OpenAI model ID (e.g. "gpt-4o", "o3", "o4-mini")
        json_output: If True, request JSON output format.
        response_schema: Optional JSON schema dict for structured output.
    """
    kwargs = dict(
        model=model_id,
        messages=[{"role": "user", "content": build_openai_content(prompt, video_input, img_input)}],
    )

    # reasoning_effort is only accepted by OpenAI reasoning models (o-series,
    # gpt-5). Chat models (gpt-4o, gpt-4.1, ...) reject it with a 400, so gate on
    # the model family rather than sending it unconditionally.
    if model_id.startswith(("o1", "o3", "o4", "gpt-5")):
        kwargs["reasoning_effort"] = thinking_level.lower()

    if json_output:
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": response_schema},
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

    return await _key_pool.call(_create, **kwargs)


async def _create(client, **kwargs):
    return GPTResponse(await client.chat.completions.create(**kwargs))
