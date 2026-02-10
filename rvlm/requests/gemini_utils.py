import json
import textwrap
import time
import io
from PIL import Image
import numpy as np

from google import genai
from google.genai import types
from google.genai.errors import ServerError

try:
    gemini_robotics = True
    if gemini_robotics:
        MODEL_ID = "gemini-robotics-er-1.5-preview"
        # load secret from /home/memmelma/Projects/vla_rl/reward_vlm/rvlm/requests/secret
        from rvlm.requests.secret import GOOGLE_API_KEY
    else:
        # gemini 3 requires paid account
        MODEL_ID = "gemini-3-flash-preview"
        from rvlm.requests.secret import GOOGLE_API_KEY

    client = genai.Client(api_key=GOOGLE_API_KEY)

except Exception as e:
    print(f"WARNING when initializing gemini utils: {e}")

def parse_json(json_output):
  # Parsing out the markdown fencing
  lines = json_output.splitlines()
  for i, line in enumerate(lines):
    if line == "```json":
      # Remove everything before "```json"
      json_output = "\n".join(lines[i + 1 :])
      # Remove everything after the closing "```"
      json_output = json_output.split("```")[0]
      break  # Exit the loop once "```json" is found
  return json_output

def img_to_mime(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img).convert("RGB")

    # PIL Image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG') # You can change format if needed
    image_bytes = img_byte_arr.getvalue()
    return types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
        )

def create_config(temperature=0.5, thinking_budget=0, include_thoughts=False):
    return types.GenerateContentConfig(
        temperature=temperature,
        thinking_config=types.ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=include_thoughts,  # Must be True to stream thinking chunks
        ),
    )

# def call_gemini_robotics_er(img, prompt, config=None):
#     if config is None:
#         config = create_config()

#     image_response = client.models.generate_content(
#         model=MODEL_ID,
#         contents=[img, prompt],
#         config=config,
#     )
#     return parse_json(image_response.text)

def call_gemini_robotics_er(img, prompt, config=None, max_retries=5, initial_delay=2.0, max_delay=60.0):
    if config is None:
        config = create_config()

    delay = initial_delay
    last_exc = None
    for attempt in range(max_retries):
        try:
            image_response = client.models.generate_content(
                model=MODEL_ID,
                contents=[img, prompt],
                config=config,
            )
            return parse_json(image_response.text)
        except ServerError as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
    if last_exc is not None:
        raise last_exc
        
async def call_gemini_robotics_er_async(img, prompt, config=None):
    if config is None:
        config = create_config()

    image_response = await client.aio.models.generate_content(
        model=MODEL_ID,
        contents=[img, prompt],
        config=config,
    )
    print(image_response.text)
    return parse_json(image_response.text)

def call_gemini_robotics_er_streaming(img, prompt, config=None, verbose=True):
    """
    Streaming version with timing info.
    
    Note: To see thinking chunks, use create_config(..., include_thoughts=True).
    Without include_thoughts, first chunk arrives only after thinking completes.
    """
    if config is None:
        config = create_config()

    start = time.time()
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] Sending request to {MODEL_ID}...")

    response_stream = client.models.generate_content_stream(
        model=MODEL_ID,
        contents=[img, prompt],
        config=config,
    )

    full_text = ""
    first_chunk = True
    first_output = True
    for chunk in response_stream:
        now = time.time()
        
        # Check if this chunk contains thinking vs output
        is_thinking = False
        if hasattr(chunk, 'candidates') and chunk.candidates:
            for candidate in chunk.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            is_thinking = True
        
        if first_chunk:
            if verbose:
                label = "queue wait" if is_thinking else "queue + thinking"
                print(f"[{time.strftime('%H:%M:%S')}] First chunk after {now - start:.1f}s ({label})")
            first_chunk = False
        
        if is_thinking and verbose:
            print("T", end="", flush=True)
        
        if chunk.text:
            if first_output and verbose:
                print(f"\n[{time.strftime('%H:%M:%S')}] Output starts at {now - start:.1f}s")
                first_output = False
            full_text += chunk.text

    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] Done in {time.time() - start:.1f}s")

    return parse_json(full_text)