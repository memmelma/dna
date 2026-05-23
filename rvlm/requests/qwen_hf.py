"""Local Hugging Face inference for Qwen3-VL (e.g. Qwen3-VL-8B-Instruct)."""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import threading
from typing import Any

import imageio
import numpy as np
import torch
from PIL import Image


def _qwen_model_classes():
    """Import lazily: Qwen3-VL needs a recent transformers (e.g. >=4.57); vlac pins <4.52."""
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as e:
        raise ImportError(
            "Qwen3-VL HF needs transformers with Qwen3VLForConditionalGeneration "
            "(typically >=4.57). This env may use an older pin (e.g. robometer[vlac]). "
            "Upgrade transformers or use robometer[robometer] / a separate venv."
        ) from e
    return AutoProcessor, Qwen3VLForConditionalGeneration


class QwenHFResponse:
    """Matches Gemini/GPT: callers use ``response.text``.
    ``response.thinking`` holds the raw ``<think>…</think>`` block (empty string if absent).
    """

    def __init__(self, text: str, thinking: str = ""):
        self.text = text
        self.thinking = thinking


_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_LOAD_LOCK = threading.Lock()


def _resolve_hf_model_id(model_id: str) -> str:
    aliases = {
        "qwen3-vl-8b-instruct": "Qwen/Qwen3-VL-8B-Instruct",
        # "qwen3-vl-8b-thinking": "Qwen/Qwen3-VL-8B-Thinking",
        # "qwen3-vl-32b-instruct": "Qwen/Qwen3-VL-32B-Instruct",
        # "qwen3-vl-32b-thinking": "Qwen/Qwen3-VL-32B-Thinking",
    }
    key = model_id.strip().lower()
    if key in aliases:
        return aliases[key]
    raw = model_id.strip()
    if "/" in raw:
        return raw
    return "Qwen/Qwen3-VL-8B-Instruct"


def _max_new_tokens(thinking_level: str) -> int:
    return {"MINIMAL": 128, "LOW": 256, "MEDIUM": 512, "HIGH": 1024}.get(
        thinking_level.upper(), 512
    )


# Qwen3-VL model card (VL vs text). ``presence_penalty`` is not in HF ``generate()``.
_VL_GEN = dict(
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.0,
    max_output_length=16384,
)
_TEXT_GEN = dict(
    do_sample=True,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    repetition_penalty=1.0,
    max_output_length=32768,
)


def _generation_kwargs(
    *,
    video_input,
    img_input,
    thinking_level: str,
) -> dict[str, Any]:
    budget = _max_new_tokens(thinking_level)
    is_vl = video_input is not None or img_input is not None
    spec = _VL_GEN if is_vl else _TEXT_GEN
    max_new = min(budget, spec["max_output_length"])
    return {
        "do_sample": spec["do_sample"],
        "temperature": spec["temperature"],
        "top_p": spec["top_p"],
        "top_k": spec["top_k"],
        "repetition_penalty": spec["repetition_penalty"],
        "max_new_tokens": max_new,
    }


def _load_model(hf_model_id: str) -> tuple[Any, Any]:
    with _LOAD_LOCK:
        if hf_model_id not in _MODEL_CACHE:
            proc_cls, model_cls = _qwen_model_classes()
            kwargs: dict[str, Any] = dict(
                pretrained_model_name_or_path=hf_model_id,
                torch_dtype="auto",
                device_map="auto",
            )
            try:
                kwargs["attn_implementation"] = "flash_attention_2"
                model = model_cls.from_pretrained(**kwargs)
            except Exception:
                kwargs.pop("attn_implementation", None)
                model = model_cls.from_pretrained(**kwargs)
            
            processor = proc_cls.from_pretrained(hf_model_id, trust_remote_code=True)
            model.eval()
            _MODEL_CACHE[hf_model_id] = (model, processor)
        return _MODEL_CACHE[hf_model_id]


def _video_path_from_input(video_input) -> str:
    if isinstance(video_input, np.ndarray):
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        imageio.mimwrite(path, video_input, format="mp4", fps=1, macro_block_size=1)
        return path
    if isinstance(video_input, str) and os.path.exists(video_input):
        return os.path.abspath(video_input)
    if isinstance(video_input, (bytes, bytearray)):
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(video_input)
        return path
    raise TypeError(f"Unsupported video_input type: {type(video_input)}")


def _pil_from_img(img) -> Image.Image:
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, str) and os.path.exists(img):
        return Image.open(img).convert("RGB")
    if isinstance(img, bytes):
        return Image.open(io.BytesIO(img)).convert("RGB")
    return img


def _build_user_content(
    prompt: str,
    video_input,
    img_input,
    video_fps: float,
) -> list[dict]:
    content: list[dict] = []
    tmp_video: str | None = None
    if video_input is not None:
        tmp_video = _video_path_from_input(video_input)
        content.append({"type": "video", "video": tmp_video, "fps": video_fps})
    if img_input is not None:
        imgs = img_input if isinstance(img_input, list) else [img_input]
        for im in imgs:
            pil = _pil_from_img(im)
            content.append({"type": "image", "image": pil})
    text = prompt
    content.append({"type": "text", "text": text})
    return content, tmp_video


def _generate_sync(
    prompt: str,
    video_input,
    img_input,
    thinking_level: str,
    hf_model_id: str,
    video_fps: float,
) -> tuple[str, str]:
    # Qwen3-VL Thinking edition: the chat template auto-injects <think> start; just
    # pass the prompt as-is.  Instruct edition does not think regardless of /think.
    effective_prompt = prompt

    content, tmp_video = _build_user_content(effective_prompt, video_input, img_input, video_fps)
    messages = [{"role": "user", "content": content}]
    try:
        model, processor = _load_model(hf_model_id)
        device = next(model.parameters()).device

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        gen_kw = _generation_kwargs(
            video_input=video_input,
            img_input=img_input,
            thinking_level=thinking_level,
        )
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kw)

        in_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, in_len:]
        out = processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        raw = out[0].strip()
        import re as _re
        # Thinking edition: template injects opening <think> before generation, so the
        # output starts with the thinking content followed by </think>\n<answer>.
        # Instruct edition: no <think> block at all.
        think_match = _re.search(r"<think>(.*?)</think>", raw, flags=_re.DOTALL)
        if think_match:
            thinking_text = think_match.group(1).strip()
            answer = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        else:
            # Thinking model: output starts directly with thinking content, ends with </think>
            orphan_close = _re.search(r"(.*?)</think>\s*", raw, flags=_re.DOTALL)
            if orphan_close:
                thinking_text = orphan_close.group(1).strip()
                answer = raw[orphan_close.end():].strip()
            else:
                thinking_text = ""
                answer = raw
        return answer, thinking_text
    finally:
        if tmp_video and tmp_video.startswith(tempfile.gettempdir()):
            try:
                os.remove(tmp_video)
            except OSError:
                pass


async def call_qwen_hf(
    prompt,
    video_input=None,
    img_input=None,
    thinking_level: str = "MEDIUM",
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    video_fps: float = 1.0,
    json_output: bool = False,
    response_schema=None,
    enable_thinking: bool = False,
    include_thoughts: bool = False,
):
    """
    Run Qwen3-VL via Hugging Face ``transformers`` (local GPU/CPU).

    Thinking is auto-detected from the resolved model id: Qwen3-VL-*-Thinking models
    emit a thinking block which is stripped from ``.text`` and stored in ``.thinking``.
    Instruct models do not think regardless of ``enable_thinking``.
    """
    _ = (json_output, response_schema, include_thoughts, enable_thinking)
    hf_id = _resolve_hf_model_id(model_id)
    result = await asyncio.to_thread(
        _generate_sync,
        prompt,
        video_input,
        img_input,
        thinking_level,
        hf_id,
        video_fps,
    )
    answer, thinking = result
    return QwenHFResponse(answer, thinking)
