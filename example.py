#!/usr/bin/env python3
"""Minimal end-to-end example: download a video from a URL and predict per-frame
task progress on the command line.

Usage:
    # defaults (the spiderman clip below, via Gemini)
    uv run python example.py

    # a specific video / task / model
    uv run python example.py \
        --url https://.../clip.mp4 \
        --task "put the spiderman in the lunch box" \
        --model openrouter/openai/gpt-5.6-terra \
        --method naive

See the README for supported model ids. For OpenRouter models, set the
OPENROUTER_API_KEY environment variable; for the direct API backends, fill in
rvlm/secrets.py.
"""

import argparse
import io
import urllib.request

import numpy as np

from rvlm import DNA

DEFAULT_URL = "https://peek-robot.github.io/static/videos/3dda+peek/8x_small_spiderman_clutter_2x_0.mp4"
DEFAULT_TASK = "put the spiderman in the lunch box"


def load_video(url: str, n_frames: int) -> np.ndarray:
    """Download an mp4 from ``url`` and uniformly subsample to ``n_frames``.

    Frame-decomposing backends (GPT, Claude, OpenRouter) send one image per
    frame, so a long clip must be subsampled to keep the request tractable.
    """
    import imageio.v3 as iio

    print(f"downloading {url}")
    data = urllib.request.urlopen(url).read()
    frames = iio.imread(io.BytesIO(data), index=None, extension=".mp4")

    if n_frames and len(frames) > n_frames:
        idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
        frames = frames[idx]
    return np.ascontiguousarray(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help="video URL (mp4)")
    parser.add_argument("--task", default=DEFAULT_TASK, help="natural-language task description")
    parser.add_argument("--model", default="openrouter/openai/gpt-5.6-terra", help="model id (see README)")
    parser.add_argument("--method", default="dna", choices=["dna", "decompose", "naive"], help="pipeline")
    parser.add_argument("--thinking", default="MEDIUM", help="reasoning effort: OFF/LOW/MEDIUM/HIGH")
    parser.add_argument("--n-frames", type=int, default=8, help="N frames to sample (0 = keep all), works best with N\in[8,20]")
    args = parser.parse_args()

    frames = load_video(args.url, args.n_frames)
    print(f"loaded {len(frames)} frames of shape {frames.shape[1:]}")
    print(f"task:   {args.task!r}")
    print(f"model:  {args.model}  (method={args.method}, thinking={args.thinking})")
    print("computing progress ...")

    dna = DNA(model=args.model, method=args.method, thinking=args.thinking, terminal_logging=True)
    progress = dna.compute_progress(frames, args.task)

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("\nper-frame progress (0 = not started, 1 = complete):")
    print(np.asarray(progress))
    print(f"\nfinal progress: {progress[-1]:.2f}   (max {progress.max():.2f})")


if __name__ == "__main__":
    main()
