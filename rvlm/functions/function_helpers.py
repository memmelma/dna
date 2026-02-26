import os
import io
import re
import json
import imageio
import numpy as np
from google import genai
from google.genai import types

GOOGLE_API_KEY = "AIzaSyCu84vRyQrfMnCkVqxpOydodfEfobPjWO0"
_client = genai.Client(api_key=GOOGLE_API_KEY)

from rvlm.annotator.annotator_reward import SimpleCoTrackerDTW
_tracker: SimpleCoTrackerDTW = None

def _get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = SimpleCoTrackerDTW()
    return _tracker

def _parse_json(text):
    if isinstance(text, list):
        return json.dumps(text)

    if not text or not text.strip():
        raise ValueError("Cannot parse empty Gemini response as JSON")

    # 1. Try to extract from ```json ... ``` code fence
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                print(f"[_parse_json] code fence content is invalid JSON ({e}), trying full text")

    # 2. Fall back to parsing the full text directly
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse Gemini response as JSON: {e}\nRaw text:\n{repr(text)}")

async def call_gemini(prompt, video_input=None, img_input=None, thinking_level="MEDIUM"):
    """
    Calls Gemini with a video, an image, and a text prompt.
    
    Args:
        video_input: Path to .mp4 file OR base64/bytes data
        img_input: Path to .jpg/.png file OR base64/bytes data
        prompt: String text
    """
    model_id = "gemini-3-flash-preview"

    parts = []

    # (optional) process video
    if video_input is not None:
        if isinstance(video_input, np.ndarray):
            buf = io.BytesIO()
            imageio.mimwrite(buf, video_input, format="mp4", fps=1)
            video_data = buf.getvalue()
        elif isinstance(video_input, str) and os.path.exists(video_input):
            with open(video_input, "rb") as f:
                video_data = f.read()
        else:
            video_data = video_input
        parts.append(types.Part.from_bytes(mime_type="video/mp4", data=video_data))

    # (optional) process image
    if img_input is not None:
        if isinstance(img_input, np.ndarray):
            buf = io.BytesIO()
            imageio.imwrite(buf, img_input, format="JPEG")
            img_data = buf.getvalue()
        elif isinstance(img_input, str) and os.path.exists(img_input):
            with open(img_input, "rb") as f:
                img_data = f.read()
        else:
            img_data = img_input
        parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=img_data))

    # process prompt
    parts.append(types.Part.from_text(text=prompt))

    # configuration
    if model_id == "gemini-3-flash-preview":
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level, include_thoughts=True),
            # thinking_config=types.ThinkingConfig(thinking_level="HIGH", include_thoughts=True),
            media_resolution="MEDIA_RESOLUTION_HIGH",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
    # elif model_id == "gemini-2.5-flash" or model_id == "gemini-2.5-flash-lite":
    #     generate_content_config = types.GenerateContentConfig(
    #         media_resolution="MEDIA_RESOLUTION_HIGH",
    #         automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    #     )
    else:
        raise ValueError(f"Unsupported model ID: {model_id}")
    # request
    response = await _client.aio.models.generate_content(
        model=model_id,
        contents=[types.Content(role="user", parts=parts)],
        config=generate_content_config,
    )

    return response