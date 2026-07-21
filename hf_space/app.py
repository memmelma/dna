#!/usr/bin/env python3
"""Gradio app (HuggingFace Space) for reward-vlm / ``dna.DNA``.

Pick an example clip (or upload a video) plus a natural-language task, choose a
model and method, and it returns:

  * an annotated **video** — the input clip on top, an animated per-frame
    progress plot in [0, 1] below it, and
  * ALL the intermediate text the VLM produced — grounded objects, per-frame
    scene descriptions, and (for ``dna_feedback``) failure / feedback — plus the
    raw per-frame progress numbers.

Why this app calls ``DNA._sample_once`` directly instead of the public
``compute_progress``: ``compute_progress`` returns *only* the averaged progress
array and discards the model's text (it is merely printed to stdout by the
``verbose`` logger). ``_sample_once`` returns the full per-sample dict
(``progress`` / ``objects`` / ``description`` / ``failure`` / ``feedback``),
which is exactly the text we want to surface. It also already handles retries,
length validation, and the 0-100 -> [0, 1] rescale.
"""

import asyncio
import contextlib
import tempfile
import traceback
from io import BytesIO

import gradio as gr
import numpy as np
from PIL import Image

from dna import DNA

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
import os

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")

DEFAULT_TASK = "put the spiderman in the lunch box"
# Default to an OpenRouter model so the Space's OPENROUTER_API_KEY secret works
# zero-config (no per-user key needed for the examples to run). Must be one of
# MODEL_CHOICES below.
DEFAULT_MODEL = "openrouter/openai/gpt-5.6-sol"

# Method dropdown: (display label -> DNA method id). The router keys off the id.
METHOD_CHOICES = [
    ("D&A", "dna"),
    ("D&A (feedback)", "dna_feedback"),
    ("Naive", "naive"),
]

THINKING_LEVELS = ["OFF", "LOW", "MEDIUM", "HIGH"]

# Model dropdown: (display label -> model id). Prefix decides the backend.
MODEL_CHOICES = [
    ("Gemini 3 Flash Preview", "gemini-3-flash-preview"),
    ("Gemini 3.5 Flash", "gemini-3.5-flash"),
    ("Gemini 3 Pro", "gemini-3-pro"),
    ("GPT-5.6 Sol", "gpt-5.6-sol"),
    ("GPT-5.6 Terra", "gpt-5.6-terra"),
    ("GPT-5.6 Sol (OpenRouter)", "openrouter/openai/gpt-5.6-sol"),
    ("GPT-5.6 Terra (OpenRouter)", "openrouter/openai/gpt-5.6-terra"),
    ("GPT-5.6 Luna (OpenRouter)", "openrouter/openai/gpt-5.6-luna"),
    ("Claude Opus 4.8 (OpenRouter)", "openrouter/anthropic/claude-opus-4.8"),
    ("Gemini 3.6 Flash (OpenRouter)", "openrouter/google/gemini-3.6-flash"),
    ("Gemini 3.5 Flash (OpenRouter)", "openrouter/google/gemini-3.5-flash"),
    ("Gemini 3 Flash Preview (OpenRouter)", "openrouter/google/gemini-3-flash-preview"),
    ("Gemini 3 Pro (OpenRouter)", "openrouter/google/gemini-3-pro-preview"),
]

# Example clips shipped in examples/ — (video filename, task instruction).
# Note: peek-robot hosts only one spiderman clip; the other two are stand-in
# manipulation clips with clear tasks (swap freely).
EXAMPLES = [
    ["spiderman_0.mp4", "put the spiderman in the lunch box"],
    ["cube_move.mp4", "pick up the red cube"],
    ["drawer.mp4", "open the drawer"],
]

# Serializes the (swap key pool -> run pipeline -> restore) window. dna's key
# pools are process-global module state, so on a shared Space two concurrent
# requests must not interleave their swaps or one user's request would run under
# another user's key. Created lazily on first use (needs a running event loop).
_run_lock: asyncio.Lock | None = None


def _get_run_lock() -> asyncio.Lock:
    global _run_lock
    if _run_lock is None:
        _run_lock = asyncio.Lock()
    return _run_lock


# ----------------------------------------------------------------------------
# Video loading — same subsample strategy as example.py
# ----------------------------------------------------------------------------
def load_frames(video_path: str | None, n_frames: int) -> np.ndarray:
    """Load frames from a video filepath (upload or example) and subsample.

    Frame-decomposing backends send one image per frame, so a long clip must be
    subsampled to keep the request tractable (``n_frames == 0`` keeps every frame).
    """
    import imageio.v3 as iio

    if not video_path:
        raise ValueError("Please choose an example or upload a video first.")
    frames = iio.imread(video_path, index=None)

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError(f"Expected (N, H, W, 3) video frames, got shape {frames.shape}")
    frames = frames[..., :3]  # drop any alpha channel

    if n_frames and len(frames) > n_frames:
        idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
        frames = frames[idx]
    return np.ascontiguousarray(frames.astype(np.uint8))


# ----------------------------------------------------------------------------
# Per-request API-key injection
# ----------------------------------------------------------------------------
def _provider_for(model_id: str) -> str:
    """Map a model id to its provider, matching dna.requests.api.call_api."""
    m = model_id.lower()
    if model_id.startswith("openrouter/"):
        return "openrouter"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("claude"):
        return "claude"
    if m.startswith("gpt") or m.startswith(("o1", "o3", "o4")):
        return "gpt"
    if m.startswith("muse"):
        return "muse"
    if m.startswith("qwen") or model_id.startswith("Qwen/"):
        return "qwen"
    raise ValueError(f"Unrecognized model id: {model_id!r}")


def _build_single_key_pool(provider: str, key: str):
    """Build a fresh single-key ``KeyPool`` for ``provider`` from a UI-typed key.

    Mirrors how each provider module constructs its own pool (client type, name,
    jitter, fail_fast) so behaviour matches upstream — just scoped to one key.
    """
    from dna.requests._pool import KeyPool

    if provider == "openrouter":
        import openai
        from dna.requests import openrouter
        return KeyPool(
            [openai.AsyncOpenAI(api_key=key, base_url=openrouter.OPENROUTER_BASE_URL)],
            name="OpenRouterKeyPool",
            empty_error="No OpenRouter API key configured.",
        )
    if provider == "gemini":
        from google import genai
        return KeyPool(
            [genai.Client(api_key=key)],
            name="GeminiKeyPool",
            empty_error="No Google API key configured.",
            jitter=(1, 3),
            fail_fast_on_400=False,
        )
    if provider == "claude":
        import anthropic
        return KeyPool(
            [anthropic.AsyncAnthropic(api_key=key)],
            name="ClaudeKeyPool",
            empty_error="No Anthropic API key configured.",
            fail_fast_on_400=False,
        )
    if provider == "gpt":
        import openai
        return KeyPool(
            [openai.AsyncOpenAI(api_key=key)],
            name="GPTKeyPool",
            empty_error="No OpenAI API key configured.",
        )
    if provider == "muse":
        return KeyPool([key], name="MuseKeyPool", empty_error="No Meta API key configured.")
    if provider == "qwen":
        raise ValueError("Qwen-VL runs locally on GPU and is not supported in this Space.")
    raise ValueError(f"Unknown provider: {provider!r}")


@contextlib.contextmanager
def use_user_key(model_id: str, api_key: str):
    """Temporarily route ``model_id``'s provider through a UI-supplied key.

    dna builds each provider's ``KeyPool`` from ``dna.secrets`` at import time,
    so a UI key can't reach it via an env var afterwards. We swap the relevant
    module-level pool for a single-key pool for the duration of the request and
    **always restore the original on exit**. This matters on a shared HF Space,
    which is a single process serving all users:

    - A blank key performs no swap, so any key already configured via
      ``secrets.py`` / the ``OPENROUTER_API_KEY`` env var still applies.
    - Restoring on exit means one user's pasted key never lingers to shadow the
      env key (or another user's key) on a later request.

    Callers must additionally serialize the swap+request window (see ``_run_lock``)
    so a concurrent request can't observe the swapped-in pool. Reaching into
    private module state is deliberate: it keeps the vendored ``dna`` package
    byte-for-byte identical to upstream (easy to re-sync).

    Yields the resolved provider name.
    """
    provider = _provider_for(model_id)
    key = (api_key or "").strip()
    if provider == "qwen":
        raise ValueError("Qwen-VL runs locally on GPU and is not supported in this Space.")
    if not key:
        yield provider  # keep whatever secrets.py / env already configured
        return

    pool = _build_single_key_pool(provider, key)

    # (module, attribute, is_dict_keys) — how to reach each provider's live pool.
    if provider == "openrouter":
        from dna.requests import openrouter as mod
        attr, dict_keys = "_key_pool", None
    elif provider == "gemini":
        from dna.requests import gemini as mod
        attr, dict_keys = "_pools", ("standard", "preview")
    elif provider == "claude":
        from dna.requests import claude as mod
        attr, dict_keys = "_key_pool", None
    elif provider == "gpt":
        from dna.requests import gpt as mod
        attr, dict_keys = "_key_pool", None
    else:  # muse
        from dna.requests import meta as mod
        attr, dict_keys = "_key_pool", None

    if dict_keys is None:
        original = getattr(mod, attr)
        setattr(mod, attr, pool)
        try:
            yield provider
        finally:
            setattr(mod, attr, original)
    else:
        # gemini keeps a {"standard"/"preview": pool} dict, built lazily.
        pools = getattr(mod, attr)
        saved = {k: pools.get(k, "__missing__") for k in dict_keys}
        for k in dict_keys:
            pools[k] = pool
        try:
            yield provider
        finally:
            for k, v in saved.items():
                if v == "__missing__":
                    pools.pop(k, None)
                else:
                    pools[k] = v


# ----------------------------------------------------------------------------
# Output formatting
# ----------------------------------------------------------------------------
def _fmt_progress_array(progress: np.ndarray) -> str:
    with np.printoptions(precision=3, suppress=True, linewidth=200):
        return str(np.asarray(progress))


def format_text(task, model, method, thinking, frames, samples, mean) -> str:
    """Render every intermediate + final text output as Markdown."""
    n_samples = len(samples)
    lines = [
        f"### Run",
        f"- **task:** `{task}`",
        f"- **model:** `{model}`  ·  **method:** `{method}`  ·  **thinking:** `{thinking}`",
        f"- **frames:** {len(frames)}  ·  **samples:** {n_samples}",
    ]

    for s_idx, s in enumerate(samples):
        lines.append("\n---")
        lines.append(f"#### Sample {s_idx + 1}/{n_samples}")

        objects = s.get("objects")
        if objects is not None:
            lines.append(f"\n**Grounded objects ({len(objects)}):**")
            lines.append("".join(f"\n- {o}" for o in objects) or "\n- _(none)_")

        description = s.get("description")
        if description is not None:
            items = description.items() if isinstance(description, dict) else enumerate(description)
            lines.append("\n**Per-frame descriptions:**")
            for i, d in items:
                lines.append(f"\n- **[{i}]** {d}")

        if s.get("success_criteria"):
            lines.append(f"\n**Success criteria (full-completion state):** {s['success_criteria']}")

        if s.get("failure"):
            lines.append(f"\n**Failure:** {s['failure']}")
        if s.get("feedback"):
            lines.append(f"\n**Feedback:** {s['feedback']}")

        prog = np.asarray(s["progress"])
        lines.append(f"\n**Per-frame progress:** `{_fmt_progress_array(prog)}`")
        if len(prog):
            lines.append(f"\n**Final:** {prog[-1]:.3f}  ·  **Max:** {prog.max():.3f}")

    if n_samples > 1:
        lines.append("\n---")
        lines.append(f"#### Mean across {n_samples} samples")
        lines.append(f"\n`{_fmt_progress_array(mean)}`")
        lines.append(f"\n**Final:** {mean[-1]:.3f}  ·  **Max:** {mean.max():.3f}")

    if method == "naive":
        lines.append("\n\n> _Note: the `naive` method makes a single direct call and produces no "
                     "grounded objects or per-frame descriptions._")

    return "\n".join(lines)


def _render_plot_frame(mean, xs, t, task, plot_w, plot_h, dpi):
    """Render one progress-plot frame (progress up to timestep ``t``) as RGB.

    Elapsed portion (0..t) is drawn in orange, upcoming (t..end) in muted blue,
    with a dot at the current frame — matching the reference styling. Title is
    the language instruction. Returns an (plot_h, plot_w, 3) uint8 array.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    bg, fg = "#0e1117", "#c9d1d9"
    elapsed_c, upcoming_c = "#ff8c42", "#4c9be8"
    n = len(mean)

    fig = Figure(figsize=(plot_w / dpi, plot_h / dpi), dpi=dpi, facecolor=bg)
    ax = fig.subplots()
    ax.set_facecolor(bg)

    # Full trace faint in the "upcoming" colour, then overlay the elapsed part.
    ax.plot(xs, mean, color=upcoming_c, linewidth=2, alpha=0.55)
    ax.plot(xs[: t + 1], mean[: t + 1], color=elapsed_c, linewidth=2.5)
    ax.scatter([xs[t]], [mean[t]], color=elapsed_c, s=45, zorder=5)

    ax.set_xlim(0, max(n - 1, 1))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("frame", color=fg, fontsize=9)
    ax.set_ylabel("progress", color=fg, fontsize=9)
    ax.set_title(task, color=fg, fontsize=10)
    ax.tick_params(colors=fg, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, color="#30363d", alpha=0.5)
    fig.tight_layout(pad=0.6)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=bg)
    buf.seek(0)
    img = np.asarray(Image.open(buf).convert("RGB"))
    if img.shape[:2] != (plot_h, plot_w):
        img = np.asarray(Image.fromarray(img).resize((plot_w, plot_h), Image.Resampling.LANCZOS))
    return img


def make_progress_video(frames: np.ndarray, mean: np.ndarray, task: str, fps: int = 4) -> str:
    """Write an mp4: input video on top, animated progress plot below.

    For each timestep the top pane is the input frame and the bottom pane is the
    progress curve revealed up to that frame. Returns the mp4 filepath.
    """
    import imageio.v3 as iio

    n, h, w, _ = frames.shape
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    xs = np.arange(n)
    # libx264 needs even width/height for BOTH panes and the stacked result.
    # Even width shared by both panes; even top-frame height; even plot height.
    plot_w = w - (w % 2)
    top_h = h - (h % 2)
    plot_h = (int(h * 0.55) // 2) * 2
    dpi = 100

    out_frames = []
    for t in range(n):
        plot_img = _render_plot_frame(mean, xs, t, task, plot_w, plot_h, dpi)
        top = frames[t, :top_h, :plot_w, :3]  # crop odd row/column if trimmed
        out_frames.append(np.concatenate([top, plot_img], axis=0))

    video = np.stack(out_frames, axis=0)
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    iio.imwrite(path, video, fps=fps, codec="libx264", macro_block_size=2)
    return path


# ----------------------------------------------------------------------------
# Main handler
# ----------------------------------------------------------------------------
async def run(video_path, task, model, method, thinking, n_frames, n_samples, api_key):
    if not (task or "").strip():
        raise gr.Error("Please enter a task / instruction.")
    if not (model or "").strip():
        raise gr.Error("Please select a model.")

    # Frame loading and provider validation are independent of the key pool —
    # do them (and their input-error reporting) before entering the lock.
    try:
        _provider_for(model)  # reject Qwen / unknown ids early with a clean message
        frames = load_frames(video_path, int(n_frames))
    except Exception as e:
        raise gr.Error(str(e))

    n_samples = max(1, int(n_samples))
    dna = DNA(model=model, method=method, thinking=thinking, verbose=0)

    # Serialize the swap+run window and always restore the pool afterwards so a
    # UI-typed key is scoped strictly to this request (never shadows the env key
    # or leaks into a concurrent user's request on a shared Space).
    try:
        async with _get_run_lock():
            with use_user_key(model, api_key):
                samples = await asyncio.gather(
                    *[dna._sample_once(frames, task) for _ in range(n_samples)]
                )
    except gr.Error:
        raise
    except Exception as e:
        # Surface provider errors (missing/invalid key, bad model id, rate limit,
        # the Gemini-2.5-needs-HIGH rule, …) as a readable message.
        raise gr.Error(f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=2)}")

    mean = np.mean([s["progress"] for s in samples], axis=0)
    text = format_text(task, model, method, thinking, frames, samples, mean)
    video = make_progress_video(frames, mean, task)
    return video, text


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
# Paper / code links — update these when the real URLs are ready.
ARXIV_URL = "https://arxiv.org/abs/0000.00000"
GITHUB_URL = "https://github.com/your-org/dna"

# BibTeX citation (placeholder — swap for the real entry when ready).
CITATION_BIBTEX = """@article{zhang2025peek,
  title={PEEK: Guiding and Minimal Image Representations for Zero-Shot Generalization of Robot Manipulation Policies},
  author={Zhang, Jesse and Memmel, Marius and Kim, Kevin and Fox, Dieter and Thomason, Jesse and Ramos, Fabio and B{\\i}y{\\i}k, Erdem and Gupta, Abhishek and Li, Anqi},
  journal={arXiv preprint arXiv:2509.18282},
  year={2025}
}"""

# Title + badges rendered via gr.HTML (not gr.Markdown): Gradio's Markdown drops
# external <img> tags and overrides the obsolete `align` attribute. Inline styles
# (text-align + a flexbox row) guarantee centering and side-by-side badges
# regardless of the Gradio theme CSS.
_HEADER_HTML = f"""
<div style="text-align:center; margin-bottom:0.5rem;">
  <h1 style="text-align:center; margin-bottom:0.75rem;">🧬 D&amp;A: Decoupling Description and Assessment<br>Enables VLMs as Zero-Shot Robotics Reward Models</h1>
  <div style="display:flex; justify-content:center; align-items:center; gap:0.5rem; flex-wrap:wrap;">
    <a href="{ARXIV_URL}" target="_blank"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
    <a href="{GITHUB_URL}" target="_blank"><img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  </div>
</div>
"""

_INTRO_MD = """
> **TL;DR** — Naïvely prompting a VLM for robot task progress collapses into a success/failure
> detector that hallucinates progress on failed runs. **D&A** *decouples* the job into two prompt
> stages — an objective, task-grounded **description** of scene dynamics, then an **assessment**
> against an explicit success rubric — yielding calibrated, fine-grained progress in [0, 1] that
> penalizes failures and interpolates successes. Purely prompt-based, no post-training.

Pick an example (or upload a robot/manipulation video), describe the task, and get per-frame
**progress** in [0, 1] as an annotated video, plus **all the intermediate text** the model
produced (grounded objects, per-frame descriptions, and — for D&A (feedback) — failure/feedback).
Choose a model and paste the matching API key.
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="D&A · zero-shot robotics reward models") as demo:
        gr.HTML(_HEADER_HTML)
        gr.Markdown(_INTRO_MD)

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="Video", sources=["upload"])
                task_in = gr.Textbox(label="Task / instruction", value=DEFAULT_TASK)
                model_in = gr.Dropdown(
                    label="Model", choices=MODEL_CHOICES, value=DEFAULT_MODEL,
                    info="Prefix decides the backend: openrouter/*, gemini*, gpt*/o*.",
                )
                with gr.Row():
                    method_in = gr.Dropdown(label="Method", choices=METHOD_CHOICES, value="dna")
                    thinking_in = gr.Dropdown(label="Thinking", choices=THINKING_LEVELS, value="MEDIUM")
                n_frames_in = gr.Slider(label="Frames (0 = all)", minimum=0, maximum=32, step=1, value=8)
                n_samples_in = gr.Slider(label="Samples (averaged)", minimum=1, maximum=5, step=1, value=1)
                gr.Markdown(
                    "⚠️ **Each added sample re-runs the whole pipeline** — cost and latency scale "
                    "linearly with *Samples* (e.g. 3 samples ≈ 3× the API cost)."
                )
                api_key_in = gr.Textbox(
                    label="API key", type="password",
                    placeholder="sk-or-… (OpenRouter) — leave blank to use the Space secret",
                )
                run_btn = gr.Button("Compute progress", variant="primary")

            with gr.Column(scale=1):
                video_out = gr.Video(label="Progress estimate", autoplay=True)
                # gr.Markdown ignores `label` in Gradio 6 (container=False), so
                # render a visible header above the output instead.
                gr.Markdown("### Model output (all text)")
                text_out = gr.Markdown()

        gr.Examples(
            examples=[[os.path.join(EXAMPLES_DIR, fn), task] for fn, task in EXAMPLES],
            inputs=[video_in, task_in],
            label="Examples (click to load a video + its instruction)",
        )

        run_btn.click(
            run,
            inputs=[video_in, task_in, model_in, method_in, thinking_in, n_frames_in, n_samples_in, api_key_in],
            outputs=[video_out, text_out],
        )

        gr.Markdown("### Citation")
        gr.Code(value=CITATION_BIBTEX, language=None, label="BibTeX", interactive=False)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
