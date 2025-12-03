import json
import textwrap
import time
import io
from PIL import Image
import numpy as np

from google import genai
from google.genai import types

gemini_robotics = True
if gemini_robotics:
    # somehow I have free access to gemini robotics
    MODEL_ID = "gemini-robotics-er-1.5-preview"
    # load secret from /home/memmelma/Projects/vla_rl/reward_vlm/rvlm/requests/secret
    from rvlm.requests.secret import GOOGLE_API_KEY
else:
    # gemini 3 requires paid account
    MODEL_ID = "gemini-3-pro-preview"
    GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY_PAID")

client = genai.Client(api_key=GOOGLE_API_KEY)

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

def create_config(temperature=0.5, thinking_budget=0):
    return types.GenerateContentConfig(
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        # automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=-1)
    )

def call_gemini_robotics_er(img, prompt, config=None):
    if config is None:
        config = create_config()

    image_response = client.models.generate_content(
        model=MODEL_ID,
        contents=[img, prompt],
        config=config,
    )

    # print(image_response.text)
    return parse_json(image_response.text)