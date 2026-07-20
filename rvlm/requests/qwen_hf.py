"""Local Hugging Face inference for Qwen vision-language models
(Qwen3-VL-*, Qwen3.6-27B, Qwen3.6-35B-A3B, ...).
"""

from __future__ import annotations

import asyncio
import inspect
import io
import os
import re
import tempfile
import threading
from typing import Any

import imageio
import numpy as np
import torch
from PIL import Image


def _qwen_model_classes():
    """Import lazily: Qwen VL models need a recent transformers (e.g. >=4.57); vlac pins <4.52.

    Uses the generic ``AutoModelForImageTextToText`` (rather than hardcoding
    ``Qwen3VLForConditionalGeneration``) so newer architectures that also ship a
    vision encoder but register under a different model class - e.g. Qwen3.6-27B
    (``Qwen3_5ForConditionalGeneration``) and Qwen3.6-35B-A3B
    (``Qwen3_5MoeForConditionalGeneration``) - resolve correctly from each model's
    own ``config.json`` without needing a class-name change here.
    """
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as e:
        raise ImportError(
            "Qwen VL HF needs a recent transformers with AutoModelForImageTextToText "
            "(typically >=4.57). This env may use an older pin (e.g. robometer[vlac]). "
            "Upgrade transformers or use robometer[robometer] / a separate venv."
        ) from e
    return AutoProcessor, AutoModelForImageTextToText


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
        "qwen3.6-27b": "Qwen/Qwen3.6-27B",
        "qwen3.6-35b-a3b": "Qwen/Qwen3.6-35B-A3B",
    }
    key = model_id.strip().lower()
    if key in aliases:
        return aliases[key]
    raw = model_id.strip()
    if "/" in raw:
        return raw
    return "Qwen/Qwen3-VL-8B-Instruct"


# Thinking-enabled generation needs a much larger budget than a plain instruct
# answer: Qwen3.6 (like Qwen3/Qwen3.5) reasons at length before emitting the
# JSON answer. Capping max_new_tokens too low (the old flat 128-1024 range)
# truncates mid-<think>, before the model ever reaches the JSON answer, which
# is exactly the "reasons too much, then produces invalid JSON if cut short"
# failure mode. _ANSWER_HEADROOM reserves room for the JSON answer *after* the
# thinking budget is exhausted, so truncation (if any) only ever hits the
# thinking span, never the answer.
_THINKING_TOKEN_BUDGETS: dict[str, int] = {
    "LOW": 1_024,
    "MEDIUM": 8_192,
    "HIGH": 24_576,
}
_ANSWER_HEADROOM = 1_024
# Instruct-mode (no thinking) answers are short JSON; keep the old small caps.
_NO_THINK_MAX_NEW_TOKENS: dict[str, int] = {
    "MINIMAL": 128,
    "LOW": 256,
    "MEDIUM": 512,
    "HIGH": 1024,
}


def _max_new_tokens(thinking_level: str, enable_thinking: bool) -> int:
    level = (thinking_level or "MEDIUM").strip().upper()
    if enable_thinking:
        return _THINKING_TOKEN_BUDGETS.get(level, _THINKING_TOKEN_BUDGETS["MEDIUM"]) + _ANSWER_HEADROOM
    return _NO_THINK_MAX_NEW_TOKENS.get(level, 512)


# Qwen3-VL model card (VL vs text). ``presence_penalty`` is not in HF ``generate()``.
# max_output_length must exceed the largest thinking budget + answer headroom
# (HIGH: 24_576 + 1_024 = 25_600) or HIGH-thinking runs get silently re-clamped.
_VL_GEN = dict(
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.0,
    max_output_length=32768,
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
    enable_thinking: bool,
) -> dict[str, Any]:
    budget = _max_new_tokens(thinking_level, enable_thinking)
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
                trust_remote_code=True,
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


def _apply_chat_template_safe(processor, messages: list, enable_thinking: bool) -> dict:
    """Call apply_chat_template, passing enable_thinking only if the processor's
    template actually accepts it (Qwen3/Qwen3.5/Qwen3.6 unified checkpoints toggle
    thinking this way; Qwen3-VL *-Thinking/*-Instruct checkpoints don't take the
    kwarg at all - thinking is baked into which checkpoint you loaded).
    """
    kwargs: dict[str, Any] = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    try:
        sig = inspect.signature(processor.apply_chat_template)
        if "enable_thinking" in sig.parameters:
            kwargs["enable_thinking"] = enable_thinking
    except (TypeError, ValueError):
        pass
    return processor.apply_chat_template(messages, **kwargs)


def _generate_sync(
    prompt: str,
    video_input,
    img_input,
    thinking_level: str,
    hf_model_id: str,
    video_fps: float,
    enable_thinking: bool,
) -> tuple[str, str]:
    # Qwen3-VL Thinking edition: the chat template auto-injects <think> start; just
    # pass the prompt as-is.  Instruct edition does not think regardless of /think.
    # Unified checkpoints (Qwen3, Qwen3.5, Qwen3.6) instead toggle via
    # enable_thinking, applied below in _apply_chat_template_safe.
    effective_prompt = prompt

    content, tmp_video = _build_user_content(effective_prompt, video_input, img_input, video_fps)
    messages = [{"role": "user", "content": content}]
    try:
        model, processor = _load_model(hf_model_id)
        device = next(model.parameters()).device

        inputs = _apply_chat_template_safe(processor, messages, enable_thinking)
        inputs = inputs.to(device)

        gen_kw = _generation_kwargs(
            video_input=video_input,
            img_input=img_input,
            thinking_level=thinking_level,
            enable_thinking=enable_thinking,
        )
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kw)

        in_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, in_len:]
        out = processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        raw = out[0].strip()
        # Thinking edition: template injects opening <think> before generation, so the
        # output starts with the thinking content followed by </think>\n<answer>.
        # Instruct edition: no <think> block at all.
        think_match = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
        if think_match:
            thinking_text = think_match.group(1).strip()
            answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        else:
            # Thinking model: output starts directly with thinking content, ends with </think>
            orphan_close = re.search(r"(.*?)</think>\s*", raw, flags=re.DOTALL)
            if orphan_close:
                thinking_text = orphan_close.group(1).strip()
                answer = raw[orphan_close.end():].strip()
            elif enable_thinking and "</think>" not in raw:
                # Generation exhausted max_new_tokens before closing </think>: the
                # entire output is unterminated thinking prose, not a JSON answer.
                # Surface this distinctly rather than letting callers try to
                # json.loads() the thinking text (the original "cut reasoning
                # short -> invalid JSON" failure mode this budget fix targets).
                thinking_text = raw
                answer = ""
            else:
                thinking_text = ""
                answer = raw

        print(f"\n{'='*60}", flush=True)
        print(f"[Qwen | model={hf_model_id} | {new_ids.shape[-1]} tokens generated]", flush=True)
        if thinking_text:
            print(f"--- reasoning ---\n{thinking_text}", flush=True)
        print(f"--- output ---\n{answer}", flush=True)
        print(f"{'='*60}\n", flush=True)

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
    Run Qwen VL models via Hugging Face ``transformers`` (local GPU/CPU).

    For separate Thinking/Instruct checkpoints (e.g. Qwen3-VL-*-Thinking),
    ``enable_thinking`` is auto-derived from the model id by the caller
    (see ``rvlm.requests.api.call_api``) and the checkpoint itself determines
    whether a thinking span is emitted. For unified checkpoints that toggle
    thinking via the chat template (Qwen3.6-27B, Qwen3.6-35B-A3B, ...),
    ``enable_thinking`` is actually applied here via ``apply_chat_template``.
    """
    _ = (json_output, response_schema, include_thoughts)
    hf_id = _resolve_hf_model_id(model_id)
    result = await asyncio.to_thread(
        _generate_sync,
        prompt,
        video_input,
        img_input,
        thinking_level,
        hf_id,
        video_fps,
        enable_thinking,
    )
    answer, thinking = result
    return QwenHFResponse(answer, thinking)
