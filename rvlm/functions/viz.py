"""Visualization helpers for logging progress predictions."""

from io import BytesIO

import numpy as np
from PIL import Image


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
