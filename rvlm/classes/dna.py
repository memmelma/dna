import asyncio
import json
from typing import Optional

import numpy as np

from rvlm.functions.description import get_cached_objects
from rvlm.functions.terminal_log import log_run
from rvlm.functions.video_to_progress import (
    get_description_from_video,
    get_description_from_video_grounded,
    get_progress_from_description,
    get_progress_from_video_naive,
)


class DNA:
    """Minimal, plug-and-play VLM progress estimator.

    Given a video of a robot performing a task and a natural-language task
    description, ``DNA`` returns a per-frame progress signal in ``[0, 1]``
    (0 = task not started, 1 = task complete). This is intended as a reward
    signal for robot learning.

    Three methods trade accuracy for cost:

    - ``"dna"``       grounding -> description -> progress (most accurate, 3 API calls)
    - ``"decompose"`` description -> progress            (faster, 2 API calls)
    - ``"naive"``     direct video -> progress           (fastest, 1 API call)

    Example::

        import numpy as np
        from rvlm import DNA

        frames = np.load("rollout.npy")            # (N, H, W, 3) uint8
        dna = DNA(model="gemini-3-flash-preview", method="dna")
        progress = dna.compute_progress(frames, "pick up the red block")
    """

    _METHOD_ALIASES = {
        "dna": "dna",
        "video_grounded_hierarchy_single": "dna",
        "decompose": "decompose",
        "video_hierarchy_single": "decompose",
        "naive": "naive",
        "video_endtoend_single": "naive",
    }

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        method: str = "dna",
        thinking: str = "MEDIUM",
        retries: int = 3,
        terminal_logging: bool = False,
    ):
        """
        Args:
            model: Model ID. Supported backends (by prefix):
                gemini*, gpt*/o*, claude*, muse*, qwen*/Qwen/*. Any model can
                also be routed through OpenRouter (handy for debugging GPT /
                Claude) with an "openrouter/<vendor>/<model>" id, e.g.
                "openrouter/openai/gpt-5.6-terra" or
                "openrouter/anthropic/claude-opus-4.8".
            method: "dna", "decompose", or "naive" (or the equivalent full
                modality strings).
            thinking: Reasoning effort: "LOW", "MEDIUM", or "HIGH".
            retries: Number of attempts before raising on failure.
            terminal_logging: If True, print the generated intermediate text
                (grounded objects, per-frame descriptions, final prediction) to
                stdout on each call.
        """
        if method not in self._METHOD_ALIASES:
            valid = sorted(set(self._METHOD_ALIASES))
            raise ValueError(f"Unknown method {method!r}. Valid options: {valid}")

        self.model = model
        self.method = self._METHOD_ALIASES[method]
        self.thinking = thinking
        self.retries = max(1, int(retries))
        self.terminal_logging = terminal_logging

    async def _run(self, frames: np.ndarray, task: str) -> tuple:
        """Run the configured pipeline. Returns (progress, objects, description).

        ``objects`` and ``description`` are None for methods that don't produce
        them; they're surfaced only so the caller can log them.
        """
        if self.method == "naive":
            results = await get_progress_from_video_naive(
                frames, task, model_id=self.model, thinking_level=self.thinking
            )
            return json.loads(results.text)[0]["progress"], None, None

        if self.method == "decompose":
            description_raw = await get_description_from_video(
                frames, task, model_id=self.model, thinking_level=self.thinking
            )
            objects = None
        else:  # "dna"
            description_raw = await get_description_from_video_grounded(
                frames, task, model_id=self.model, thinking_level=self.thinking
            )
            objects = get_cached_objects(frames, task)

        description = {i: d["description"] for i, d in enumerate(description_raw)}
        results = await get_progress_from_description(
            description, task, model_id=self.model, thinking_level=self.thinking
        )
        return json.loads(results.text)[0]["progress"], objects, description

    async def compute_progress_async(self, frames: np.ndarray, task: str) -> np.ndarray:
        """Async variant of :meth:`compute_progress` for use inside an event loop.

        Args:
            frames: (N, H, W, 3) uint8 array of video frames.
            task: Natural-language task description.

        Returns:
            (N,) float array of per-frame progress in [0, 1].

        Raises:
            RuntimeError: if all retries fail.
        """
        last_error: Optional[Exception] = None
        for _ in range(self.retries):
            try:
                progress, objects, description = await self._run(frames, task)
                if len(progress) != len(frames):
                    raise ValueError(
                        f"Progress length {len(progress)} != frame count {len(frames)}"
                    )
                progress = np.asarray(progress, dtype=np.float64) / 100.0
                if self.terminal_logging:
                    log_run(task, objects=objects, description=description, progress=progress)
                return progress
            except Exception as e:  # noqa: BLE001 — retry on any failure
                last_error = e

        raise RuntimeError(
            f"DNA progress computation failed after {self.retries} attempt(s): {last_error}"
        ) from last_error

    def compute_progress(self, frames: np.ndarray, task: str) -> np.ndarray:
        """Compute per-frame task progress.

        Synchronous wrapper around :meth:`compute_progress_async`. Do not call
        this from inside a running event loop (e.g. Jupyter); use
        :meth:`compute_progress_async` there instead.

        Args:
            frames: (N, H, W, 3) uint8 array of video frames.
            task: Natural-language task description.

        Returns:
            (N,) float array of per-frame progress in [0, 1].
        """
        return asyncio.run(self.compute_progress_async(frames, task))
