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

    # also save the input/progress videos and a text log (verbose=2)
    uv run python example.py --n-samples 3 --log-dir logs/example

See the README for supported model ids. For OpenRouter models, set the
OPENROUTER_API_KEY environment variable; for the direct API backends, fill in
dna/secrets.py.
"""

import argparse
import io
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from dna import DNA

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
    parser.add_argument("--n-frames", type=int, default=8, help="N frames to sample (0 = keep all), works best with N in [8,20]")
    parser.add_argument("--n-samples", type=int, default=1, help="number of independent pipeline runs (averaged)")
    parser.add_argument("--log-dir", default=None,
                        help="if set, write input.mp4, progress.mp4, and log.txt under this dir (verbose=2)")
    args = parser.parse_args()

    frames = load_video(args.url, args.n_frames)
    print(f"loaded {len(frames)} frames of shape {frames.shape[1:]}")
    print(f"task:   {args.task!r}")
    print(f"model:  {args.model}  (method={args.method}, thinking={args.thinking}, n_samples={args.n_samples})")
    print("computing progress ...")

    # verbose=2 makes DNA save videos (input.mp4, progress.mp4) and a text log
    # under log_dir/<video_name>/. With n_samples > 1 the progress overlay shows
    # every sample's trace plus their mean; the log records each sample separately.
    verbose = 2 if args.log_dir else 1
    video_name = Path(urlparse(args.url).path).stem or "example"

    dna = DNA(
        model=args.model,
        method=args.method,
        thinking=args.thinking,
        verbose=verbose,
        log_dir=args.log_dir,
    )
    progress = dna.compute_progress(
        frames, args.task, n_samples=args.n_samples, video_name=video_name
    )

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("\nper-frame progress (0 = not started, 1 = complete):")
    print(np.asarray(progress))
    print(f"\nfinal progress: {progress[-1]:.2f}   (max {progress.max():.2f})")
    if args.log_dir:
        print(f"\nlogs (input.mp4, progress.mp4, log.txt) written to "
              f"{Path(args.log_dir) / video_name}")


if __name__ == "__main__":
    main()
