"""Visualization helpers for logging progress predictions."""

from io import BytesIO
from typing import Sequence

import numpy as np
from PIL import Image


def progress_video_multi(
    video: np.ndarray, progress_runs: Sequence[np.ndarray], title: str = ""
) -> np.ndarray:
    """Render side-by-side frames: RGB video + all progress traces overlaid.

    Args:
        video: (N, H, W, 3) uint8 frames.
        progress_runs: List of (N,) per-frame progress arrays in [0, 1], one per run.
        title: Figure title (e.g. the task description).

    Returns:
        (N, H, W + plot_w, 3) uint8 array — each original frame concatenated
        with the multi-trace progress plot up to that timestep.
    """
    if len(progress_runs) == 1:
        return progress_video(video, progress_runs[0], title=title)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, h, w, _ = video.shape
    runs = [np.asarray(p, dtype=np.float64).reshape(-1) for p in progress_runs]
    for i, r in enumerate(runs):
        if r.shape[0] != n:
            raise ValueError(f"progress_runs[{i}] length {r.shape[0]} != video length {n}")
    if n == 0:
        return video.copy()

    plot_w, dpi = 320, 100
    fig_w_in, fig_h_in = plot_w / dpi, h / dpi
    x_max = max(n - 1, 1)
    colors = [f"C{i}" for i in range(len(runs))]

    out = []
    for t in range(n):
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        xs = np.arange(t + 1)
        for i, run in enumerate(runs):
            ys = run[: t + 1]
            ax.plot(xs, ys, color=colors[i], linewidth=1.5, alpha=0.7, label=f"run {i+1}")
            ax.scatter([t], [run[t]], color=colors[i], s=24, zorder=5, alpha=0.8)

        mean = np.mean([r[: t + 1] for r in runs], axis=0)
        ax.plot(xs, mean, color="black", linewidth=2.5, linestyle="--", label="mean")
        ax.scatter([t], [mean[-1]], color="black", s=36, zorder=6)

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel("timestep")
        ax.set_ylabel("progress")
        ax.grid(True, alpha=0.3)
        if t == 0:
            ax.legend(fontsize=6, loc="upper left")
        fig.tight_layout(pad=0.4)
        fig.suptitle(title, fontsize=7)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white", pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        plot_img = np.asarray(Image.open(buf).convert("RGB"))
        plot_img = np.array(
            Image.fromarray(plot_img).resize((plot_w, h), Image.Resampling.LANCZOS)
        )
        out.append(np.concatenate([video[t], plot_img], axis=1))
    return np.stack(out, axis=0)


def progress_video(video: np.ndarray, progress: np.ndarray, title: str = "") -> np.ndarray:
    """Render side-by-side frames: RGB video + a progress-vs-timestep plot.

    Args:
        video: (N, H, W, 3) uint8 frames.
        progress: (N,) per-frame progress values in [0, 1].
        title: Figure title (e.g. the task description).

    Returns:
        (N, H, W + plot_w, 3) uint8 array — each original frame concatenated
        with the progress plot up to that timestep.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, h, w, _ = video.shape
    prog = np.asarray(progress, dtype=np.float64).reshape(-1)
    if prog.shape[0] != n:
        raise ValueError(f"progress length {prog.shape[0]} != video length {n}")
    if n == 0:
        return video.copy()

    plot_w, dpi = 320, 100
    fig_w_in, fig_h_in = plot_w / dpi, h / dpi
    x_max = max(n - 1, 1)

    out = []
    for t in range(n):
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        xs = np.arange(t + 1)
        ys = prog[: t + 1]
        ax.plot(xs, ys, color="C0", linewidth=2)
        ax.scatter([t], [prog[t]], color="red", s=36, zorder=5)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel("timestep")
        ax.set_ylabel("progress")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=0.4)
        fig.suptitle(title)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white", pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        plot_img = np.asarray(Image.open(buf).convert("RGB"))
        plot_img = np.array(
            Image.fromarray(plot_img).resize((plot_w, h), Image.Resampling.LANCZOS)
        )
        out.append(np.concatenate([video[t], plot_img], axis=1))
    return np.stack(out, axis=0)
