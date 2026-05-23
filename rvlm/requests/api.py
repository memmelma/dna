import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from rvlm.requests.gemini import call_gemini
from rvlm.requests.gpt import call_gpt

async def call_api(prompt, video_input=None, img_input=None, thinking_level="MEDIUM", media_resolution="MEDIA_RESOLUTION_HIGH", model_id: str = "gemini-3-flash-preview", json_output: bool = False, response_schema=None, include_thoughts=False, max_new_tokens: int | None = None, greedy: bool = False):
    if model_id.startswith("gemini"):
        # NOTE: overwriting media_resolution for Gemini models
        media_resolution="MEDIA_RESOLUTION_HIGH"
        return await call_gemini(prompt, video_input, img_input, thinking_level, model_id, media_resolution, json_output, response_schema=response_schema, include_thoughts=include_thoughts)
    elif model_id.startswith("gpt"):
        # NOTE: overwriting media_resolution for GPT models
        media_resolution="low"
        return await call_gpt(prompt, video_input, img_input, thinking_level, model_id, media_resolution, json_output, response_schema)
    elif model_id.lower().startswith("qwen") or model_id.startswith("Qwen/"):
        from rvlm.requests.qwen_hf import call_qwen_hf, _resolve_hf_model_id

        hf_id = _resolve_hf_model_id(model_id)
        enable_thinking = "thinking" in hf_id.lower()
        return await call_qwen_hf(
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