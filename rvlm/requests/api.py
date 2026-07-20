import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from rvlm.requests.gemini import call_gemini
from rvlm.requests.gpt import call_gpt
from rvlm.requests.deepseek import call_deepseek
from rvlm.requests.meta import call_muse

async def call_api(prompt, video_input=None, img_input=None, thinking_level="MEDIUM", media_resolution="MEDIA_RESOLUTION_HIGH", model_id: str = "gemini-3-flash-preview", json_output: bool = False, response_schema=None, include_thoughts=False, max_new_tokens: int | None = None, greedy: bool = False):
    if model_id.startswith("gemini"):
        # NOTE: overwriting media_resolution for Gemini models
        media_resolution="MEDIA_RESOLUTION_HIGH"
        return await call_gemini(prompt, video_input, img_input, thinking_level, model_id, media_resolution, json_output, response_schema=response_schema, include_thoughts=include_thoughts)
    elif model_id.startswith("gpt"):
        # NOTE: overwriting media_resolution for GPT models
        media_resolution="low"
        return await call_gpt(prompt, video_input, img_input, thinking_level, model_id, media_resolution, json_output, response_schema)
    elif model_id.startswith("deepseek"):
        # NOTE: overwriting media_resolution for DeepSeek models (frames-as-images, like GPT)
        media_resolution="low"
        return await call_deepseek(prompt, video_input, img_input, thinking_level, model_id, media_resolution, json_output, response_schema)
    elif model_id.startswith("muse"):
        return await call_muse(prompt, video_input, img_input, thinking_level, model_id, media_resolution, json_output, response_schema)
    elif model_id.lower().startswith("qwen") or model_id.startswith("Qwen/"):
        from rvlm.requests.qwen_hf import call_qwen_hf, _resolve_hf_model_id

        hf_id = _resolve_hf_model_id(model_id)
        hf_id_lower = hf_id.lower()
        if "thinking" in hf_id_lower:
            # Separate *-Thinking checkpoint (e.g. Qwen3-VL-8B-Thinking): always thinks.
            enable_thinking = True
        elif "instruct" in hf_id_lower:
            # Separate *-Instruct checkpoint: never thinks regardless of thinking_level.
            enable_thinking = False
        else:
            # Unified checkpoint (Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.5, ...): thinking
            # is toggled via the chat template, driven by thinking_level like Gemini/GPT.
            enable_thinking = (thinking_level or "MEDIUM").strip().upper() not in ("OFF", "NONE", "FALSE", "NO")
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