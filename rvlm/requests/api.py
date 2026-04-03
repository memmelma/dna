import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from rvlm.requests.gemini import call_gemini
from rvlm.requests.gpt import call_gpt

async def call_api(prompt, video_input=None, img_input=None, thinking_level="MEDIUM", model_id: str = "gemini-3-flash-preview", media_resolution="MEDIA_RESOLUTION_HIGH", json_output: bool = False, response_schema=None, include_thoughts=False):
    if model_id.startswith("gemini"):
        return await call_gemini(prompt, video_input, img_input, thinking_level, model_id, media_resolution, json_output, response_schema=response_schema, include_thoughts=include_thoughts)
    elif model_id.startswith("gpt"):
        return await call_gpt(prompt, video_input, img_input, thinking_level, model_id, media_resolution, json_output, response_schema)
    elif model_id.lower().startswith("qwen") or model_id.startswith("Qwen/"):
        from rvlm.requests.qwen_hf import call_qwen_hf
        return await call_qwen_hf(
            prompt,
            video_input=video_input,
            img_input=img_input,
            thinking_level=thinking_level,
            model_id=model_id,
            json_output=json_output,
            response_schema=response_schema,
        )
    else:
        raise ValueError(f"Invalid model ID: {model_id}")