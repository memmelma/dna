import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from dna.functions.description import get_cached_objects
from dna.functions.terminal_log import log_run
from dna.functions.video_to_progress import (
    get_description_from_video,
    get_description_from_video_grounded,
    get_progress_from_description,
    get_progress_from_description_failure,
    get_progress_from_video_naive,
)
from dna.functions.viz import progress_video_multi


class DNA:
    """Minimal, plug-and-play VLM progress estimator.

    Given a video of a robot performing a task and a natural-language task
    description, ``DNA`` returns a per-frame progress signal in ``[0, 1]``
    (0 = task not started, 1 = task complete). This is intended as a reward
    signal for robot learning.

    Four methods trade accuracy for cost:

    - ``"dna"``          grounding -> description -> progress (most accurate, 3 API calls)
    - ``"dna_feedback"`` like "dna", but the progress step also reports failure
                         and feedback text when the task isn't fully completed
    - ``"decompose"``    description -> progress            (faster, 2 API calls)
    - ``"naive"``        direct video -> progress           (fastest, 1 API call)

    Example::

        import numpy as np
        from dna import DNA

        frames = np.load("rollout.npy")            # (N, H, W, 3) uint8
        dna = DNA(model="gemini-3-flash-preview", method="dna")
        progress = dna.compute_progress(frames, "pick up the red block")
    """

    _METHOD_ALIASES = {
        "dna": "dna",
        "video_grounded_hierarchy_single": "dna",
        "dna_feedback": "dna_feedback",
        "video_grounded_hierarchy_failure": "dna_feedback",
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
        verbose: int = 1,
        log_dir: Optional[str] = None,
    ):
        """
        Args:
            model: Model ID. Supported backends (by prefix):
                gemini*, gpt*/o*, claude*, muse*, qwen*/Qwen/*. Any model can
                also be routed through OpenRouter (handy for debugging GPT /
                Claude) with an "openrouter/<vendor>/<model>" id, e.g.
                "openrouter/openai/gpt-5.6-terra" or
                "openrouter/anthropic/claude-opus-4.8".
            method: "dna", "dna_feedback", "decompose", or "naive" (or the
                equivalent full modality strings).
            thinking: Reasoning effort: "LOW", "MEDIUM", or "HIGH".
            retries: Number of attempts before raising on failure.
            verbose: Logging level.
                0 = silent;
                1 = terminal only (grounded objects, per-frame descriptions,
                    final prediction, per-rollout wall-clock time);
                2 = terminal plus text log (``log.txt``) and videos
                    (``input.mp4``, ``progress.mp4``) written under ``log_dir``.
            log_dir: Directory for the text / video logs written at ``verbose=2``.
                Required for those files to be saved; a per-run subdirectory is
                created inside it.
        """
        if method not in self._METHOD_ALIASES:
            valid = sorted(set(self._METHOD_ALIASES))
            raise ValueError(f"Unknown method {method!r}. Valid options: {valid}")

        self.model = model
        self.method = self._METHOD_ALIASES[method]
        self.thinking = thinking
        self.retries = max(1, int(retries))
        self.verbose = int(verbose)
        self.log_dir = Path(log_dir) if log_dir else None

    def _save_logs(
        self,
        frames: np.ndarray,
        task: str,
        samples: list,
        *,
        video_name: Optional[str] = None,
        elapsed: Optional[float] = None,
    ) -> Optional[Path]:
        """Write the text log and videos for one call (``verbose=2``).

        Requires ``log_dir``; returns None (and writes nothing) if it is unset.
        Creates a subdirectory named after the video (or a timestamp) containing:
          - ``input.mp4``:    the raw input frames as a video
          - ``progress.mp4``: side-by-side with animated progress plot
          - ``log.txt``:      text summary with model outputs

        Args:
            samples: List of per-sample dicts with keys ``progress``, ``objects``,
                ``description``, ``success_criteria``, ``failure``, ``feedback``.
            elapsed: Wall-clock seconds for the full compute.

        Returns the path to the run subdirectory, or None if ``log_dir`` is unset.
        """
        if self.log_dir is None:
            print("[dna] verbose=2 but log_dir is not set; skipping text/video logs")
            return None

        import imageio.v3 as iio

        run_name = video_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.log_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        all_progress = [s["progress"] for s in samples]

        iio.imwrite(run_dir / "input.mp4", frames, fps=4)
        overlay = progress_video_multi(frames, all_progress, title=task)
        iio.imwrite(run_dir / "progress.mp4", overlay, fps=4)

        with open(run_dir / "log.txt", "w") as f:
            f.write(f"task: {task}\n")
            f.write(f"model: {self.model}\n")
            f.write(f"method: {self.method}\n")
            f.write(f"thinking: {self.thinking}\n")
            f.write(f"n_frames: {len(frames)}\n")
            f.write(f"n_samples: {len(samples)}\n")
            f.write(f"timestamp: {datetime.now().isoformat()}\n")
            if elapsed is not None:
                f.write(f"elapsed: {elapsed:.1f}s\n")

            for s_idx, s in enumerate(samples):
                f.write(f"\n{'─' * 60}\n")
                f.write(f"sample {s_idx + 1}/{len(samples)}\n")
                f.write(f"{'─' * 60}\n")

                if s["objects"] is not None:
                    f.write(f"objects ({len(s['objects'])}):\n")
                    for obj in s["objects"]:
                        f.write(f"  {obj}\n")
                    f.write("\n")

                if s["description"] is not None:
                    desc = s["description"]
                    items = desc.items() if isinstance(desc, dict) else enumerate(desc)
                    f.write("descriptions:\n")
                    for i, d in items:
                        f.write(f"  [{i}] {d}\n")
                    f.write("\n")

                if s.get("success_criteria"):
                    f.write(f"success criteria: {s['success_criteria']}\n\n")

                if s.get("failure"):
                    f.write(f"failure: {s['failure']}\n\n")
                if s.get("feedback"):
                    f.write(f"feedback: {s['feedback']}\n\n")

                progress = s["progress"]
                f.write("progress:\n")
                for i, p in enumerate(progress):
                    f.write(f"  [{i}] {p:.4f}\n")
                f.write(f"final: {progress[-1]:.4f}  max: {progress.max():.4f}\n")

            if len(samples) > 1:
                mean = np.mean(all_progress, axis=0)
                f.write(f"\n{'─' * 60}\n")
                f.write("mean progress across samples:\n")
                for i, p in enumerate(mean):
                    f.write(f"  [{i}] {p:.4f}\n")
                f.write(f"mean final: {mean[-1]:.4f}  mean max: {mean.max():.4f}\n")

        return run_dir

    async def _run(self, frames: np.ndarray, task: str) -> dict:
        """Run the configured pipeline once.

        Returns a dict with keys ``progress`` (raw 0-100 list), ``objects``,
        ``description``, ``success_criteria`` (the model's explicit definition of
        full completion, from the prompt's "completion state" field), ``failure``,
        ``feedback``. Fields not produced by the configured method are None.
        """
        if self.method == "naive":
            results = await get_progress_from_video_naive(
                frames, task, model_id=self.model, thinking_level=self.thinking
            )
            # The naive prompt returns only "progress" (no completion state).
            return {
                "progress": json.loads(results.text)[0]["progress"],
                "objects": None,
                "description": None,
                "success_criteria": None,
                "failure": None,
                "feedback": None,
            }

        if self.method == "decompose":
            description_raw = await get_description_from_video(
                frames, task, model_id=self.model, thinking_level=self.thinking
            )
            objects = None
        else:  # "dna" / "dna_feedback"
            description_raw = await get_description_from_video_grounded(
                frames, task, model_id=self.model, thinking_level=self.thinking
            )
            objects = get_cached_objects(frames, task)

        description = {i: d["description"] for i, d in enumerate(description_raw)}

        if self.method == "dna_feedback":
            results = await get_progress_from_description_failure(
                description, task, model_id=self.model, thinking_level=self.thinking
            )
            payload = json.loads(results.text)[0]
            return {
                "progress": payload["progress"],
                "objects": objects,
                "description": description,
                "success_criteria": payload.get("completion state"),
                "failure": payload.get("failure"),
                "feedback": payload.get("feedback"),
            }

        results = await get_progress_from_description(
            description, task, model_id=self.model, thinking_level=self.thinking
        )
        payload = json.loads(results.text)[0]
        return {
            "progress": payload["progress"],
            "objects": objects,
            "description": description,
            "success_criteria": payload.get("completion state"),
            "failure": None,
            "feedback": None,
        }

    async def _sample_once(self, frames: np.ndarray, task: str) -> dict:
        """One pipeline sample with retries; returns the validated sample dict."""
        last_error: Optional[Exception] = None
        for _ in range(self.retries):
            try:
                sample = await self._run(frames, task)
                if len(sample["progress"]) != len(frames):
                    raise ValueError(
                        f"Progress length {len(sample['progress'])} != frame count {len(frames)}"
                    )
                sample["progress"] = np.asarray(sample["progress"], dtype=np.float64) / 100.0
                return sample
            except Exception as e:  # noqa: BLE001 — retry on any failure
                last_error = e

        raise RuntimeError(
            f"DNA progress computation failed after {self.retries} attempt(s): {last_error}"
        ) from last_error

    async def compute_progress_async(
        self,
        frames: np.ndarray,
        task: str,
        *,
        n_samples: int = 1,
        video_name: Optional[str] = None,
    ) -> np.ndarray:
        """Async variant of :meth:`compute_progress` for use inside an event loop.

        Args:
            frames: (N, H, W, 3) uint8 array of video frames.
            task: Natural-language task description.
            n_samples: Number of independent pipeline runs. Samples run
                concurrently and the returned progress is their mean.
            video_name: Optional name used for the log subdirectory (e.g. the
                source filename without extension). Ignored when ``log_dir`` is
                not set.

        Returns:
            (N,) float array of per-frame progress in [0, 1], averaged over
            ``n_samples`` runs.

        Raises:
            RuntimeError: if all retries fail for any sample.
        """
        n_samples = max(1, int(n_samples))

        t0 = time.perf_counter()
        samples = await asyncio.gather(
            *[self._sample_once(frames, task) for _ in range(n_samples)]
        )
        elapsed = time.perf_counter() - t0

        progress = np.mean([s["progress"] for s in samples], axis=0)

        if self.verbose >= 1:
            for s_idx, s in enumerate(samples):
                header = task if n_samples == 1 else f"{task} [sample {s_idx + 1}/{n_samples}]"
                log_run(
                    header,
                    objects=s["objects"],
                    description=s["description"],
                    success_criteria=s.get("success_criteria"),
                    progress=s["progress"],
                    failure=s.get("failure"),
                    feedback=s.get("feedback"),
                )
            if n_samples > 1:
                with np.printoptions(precision=3, suppress=True, linewidth=200):
                    print(f"\nmean progress across {n_samples} samples: {progress}")
            print(f"\n⏱  {elapsed:.1f}s ({n_samples} sample{'s' if n_samples > 1 else ''})")

        if self.verbose >= 2:
            self._save_logs(frames, task, samples, video_name=video_name, elapsed=elapsed)

        return progress

    def compute_progress(
        self,
        frames: np.ndarray,
        task: str,
        *,
        n_samples: int = 1,
        video_name: Optional[str] = None,
    ) -> np.ndarray:
        """Compute per-frame task progress.

        Synchronous wrapper around :meth:`compute_progress_async`. Do not call
        this from inside a running event loop (e.g. Jupyter); use
        :meth:`compute_progress_async` there instead.

        Args:
            frames: (N, H, W, 3) uint8 array of video frames.
            task: Natural-language task description.
            n_samples: Number of independent pipeline runs (concurrent); the
                returned progress is their mean.
            video_name: Optional name used for the log subdirectory.

        Returns:
            (N,) float array of per-frame progress in [0, 1], averaged over
            ``n_samples`` runs.
        """
        return asyncio.run(
            self.compute_progress_async(
                frames, task, n_samples=n_samples, video_name=video_name
            )
        )
