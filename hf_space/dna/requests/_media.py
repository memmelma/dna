"""Shared media encoding/decoding helpers for the request providers.

All providers accept video / image inputs in the same three forms:
  - ``np.ndarray`` — a (T,H,W,C) video or (H,W,C) image
  - ``str`` — a path to an .mp4 / .jpg / .png file
  - ``bytes`` — already-encoded media

These helpers centralize that dispatch so each provider stays a thin wrapper.
"""

import base64
import io
import os

import imageio
import numpy as np


def encode_b64(data: bytes) -> str:
    """Base64-encode bytes to an ASCII string."""
    return base64.b64encode(data).decode("utf-8")


def video_to_frames(video_input) -> np.ndarray:
    """Decode ``video_input`` into a (T,H,W,C) frame array.

    For the frame-decomposing providers (GPT, Claude, OpenRouter) that have no
    native video support and send one image per frame.
    """
    if isinstance(video_input, np.ndarray):
        return video_input
    if isinstance(video_input, str) and os.path.exists(video_input):
        return np.stack(imageio.mimread(video_input, memtest=False))
    buf = io.BytesIO(video_input if isinstance(video_input, bytes) else video_input)
    return np.stack(imageio.mimread(buf, format="mp4", memtest=False))


def video_to_bytes(video_input) -> bytes:
    """Encode ``video_input`` to raw mp4 bytes.

    For the native-video providers (Gemini, Muse) that send the whole clip.
    """
    if isinstance(video_input, np.ndarray):
        buf = io.BytesIO()
        imageio.mimwrite(buf, video_input, format="mp4", fps=1, macro_block_size=1)
        return buf.getvalue()
    if isinstance(video_input, str) and os.path.exists(video_input):
        with open(video_input, "rb") as f:
            return f.read()
    return video_input if isinstance(video_input, bytes) else bytes(video_input)


def img_to_bytes(img) -> bytes:
    """Encode a single image (ndarray / path / bytes) to raw JPEG bytes."""
    if isinstance(img, np.ndarray):
        buf = io.BytesIO()
        imageio.imwrite(buf, img, format="JPEG")
        return buf.getvalue()
    if isinstance(img, str) and os.path.exists(img):
        with open(img, "rb") as f:
            return f.read()
    return img if isinstance(img, bytes) else bytes(img)


def frames_to_jpegs(frames: np.ndarray) -> list:
    """Encode a (T,H,W,C) frame array to a list of per-frame JPEG bytes."""
    out = []
    for frame in frames:
        buf = io.BytesIO()
        imageio.imwrite(buf, frame, format="JPEG")
        out.append(buf.getvalue())
    return out


def as_image_list(img_input) -> list:
    """Normalize ``img_input`` (a single image or a list) to a list."""
    return img_input if isinstance(img_input, list) else [img_input]


def media_jpegs(video_input=None, img_input=None) -> list:
    """Collect all inputs as a flat list of JPEG bytes (video frames first).

    Shared by every frame-decomposing provider; each then wraps the bytes in
    its own content-block format.
    """
    jpegs = []
    if video_input is not None:
        jpegs.extend(frames_to_jpegs(video_to_frames(video_input)))
    if img_input is not None:
        jpegs.extend(img_to_bytes(img) for img in as_image_list(img_input))
    return jpegs


def build_openai_content(prompt, video_input=None, img_input=None, detail: str = "low") -> list:
    """Build an OpenAI-compatible message content array (images then text).

    Used by the GPT and OpenRouter providers, whose chat-completions API takes
    ``image_url`` data-URI parts followed by a text part.
    """
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_b64(jpeg)}", "detail": detail},
        }
        for jpeg in media_jpegs(video_input, img_input)
    ]
    content.append({"type": "text", "text": prompt})
    return content
