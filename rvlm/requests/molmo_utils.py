import json
import re
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# Model configuration
MODEL_ID = "allenai/Molmo2-8B"

# Global model and processor (lazy loaded)
_processor = None
_model = None


def get_model_and_processor():
    """Lazy load the model and processor."""
    global _processor, _model
    
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
        )
    
    if _model is None:
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
        )
    
    return _processor, _model


def parse_json(json_output):
    """Parsing out the markdown fencing and fix Molmo2 output format."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1:])
            json_output = json_output.split("```")[0]
            break
    
    # Fix Molmo2's space-separated arrays like [427 162] -> [427, 162]
    json_output = re.sub(r'\[(\d+)\s+(\d+)\]', r'[\1, \2]', json_output)
    
    return json_output


def img_to_pil(img):
    """Convert numpy array to PIL Image if needed."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img).convert("RGB")
    return img


def call_molmo2(img, prompt, max_new_tokens=2048):
    """
    Call Molmo2 with a single image and prompt.
    
    Args:
        img: PIL Image or numpy array
        prompt: Text prompt string
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Parsed JSON string from the model response
    """
    processor, model = get_model_and_processor()
    
    # Convert to PIL if needed
    img = img_to_pil(img)
    
    # Build messages in Molmo2 format
    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text=prompt),
                dict(type="image", image=img),
            ],
        }
    ]
    
    # Process inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate output
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Decode generated tokens
    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return parse_json(generated_text)


