import asyncio
import datetime
import json
import os
import time
from typing import List, Optional

import imageio
import numpy as np

from dna.functions.description import (
    get_cached_objects,
    get_description_from_video,
    get_description_from_video_grounded,
    get_description_from_video_grounded_extra,
)
from dna.functions.terminal_log import log_run
from dna.functions.viz import progress_video
from dna.functions.progress import (
    get_progress_from_description,
    get_progress_from_description_failure_extra,
    get_progress_from_description_roboreward,
    get_progress_from_video,
    get_progress_from_video_naive,
    get_progress_from_video_roboreward,
)


def response_to_json(response) -> dict:
    response_json = json.loads(response.text)
    try:
        return response_json[0]
    except KeyError:
        return response_json
    except Exception as e:
        raise ValueError(f"Failed to parse response: {e}")


def _extract_completion_state(text):
    """Best-effort pull of the model's "completion state" (success criteria) from
    a raw response ``text``. Returns None if absent or unparseable — logging only.
    """
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if isinstance(parsed, dict):
        return parsed.get("completion state")
    return None


class Experimental:
    """Research estimator exposing the video-input modality / processing /
    reasoning combinations. For a minimal, release-ready interface see ``DNA``.

    ``modality`` is a compound string ``{input}_{processing}_{reasoning}``:

    - input:      ``video``, ``video_grounded``, ``video_roboreward``
    - processing: ``hierarchy`` (describe then score), ``endtoend`` (score
                  directly), ``naive`` (video-only, single call)
    - reasoning:  ``single``, ``roboreward``

    Valid combinations (enforced by asserts in ``_compute_progress_once_async``):
    ``video_naive_single``, ``video_endtoend_single``,
    ``video_roboreward_endtoend_single``, ``video_grounded_hierarchy_single``,
    ``video_grounded_hierarchy_roboreward`` (and the ``video_*`` hierarchy
    variants).
    """

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        thinking_level: str = "MEDIUM",
        modality: str = "video_grounded_hierarchy_single",
        verbose: int = 1,
        log_dir: Optional[str] = None,
        extra_instruction_description: Optional[str] = None,
        extra_instruction_assess: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            verbose: Logging level.
                0 = silent;
                1 = terminal only (grounded objects, per-frame descriptions,
                    final prediction, per-rollout wall-clock time);
                2 = terminal plus text logs and videos written under ``log_dir``.
            log_dir: Base directory for the text / video logs written at
                ``verbose=2`` (defaults to ``./dna_logs``); a timestamped
                subdirectory is created inside it.
            extra_instruction_description: Global natural-language instruction
                injected into the description stage of the ``dna_extra`` modality
                (e.g. "proprioceptive state is [x, y, z, gripper open/close]").
                Ignored by other modalities.
            extra_instruction_assess: Global natural-language instruction injected
                into the progress/assessment stage of the ``dna_extra`` modality
                (e.g. "consider smoothness of the trajectory"). Ignored by other
                modalities.
        """
        self.model_name = model_name
        self.thinking_level = thinking_level
        self.modality = modality
        self.verbose = int(verbose)
        self.extra_instruction_description = extra_instruction_description
        self.extra_instruction_assess = extra_instruction_assess
        self.ctr = 0

        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.root_dir = None
        if self.verbose >= 2:
            base = log_dir or os.path.join(os.getcwd(), "dna_logs")
            self.root_dir = os.path.join(base, self.timestamp)
            os.makedirs(self.root_dir, exist_ok=True)

    async def _compute_progress_once_async(self, frames_array: np.ndarray, task_description: str = "", per_step_info: Optional[list] = None, return_details: bool = False):
        """
        One pipeline run (single sample). See compute_progress_async.

        :param frames_array: (N, H, W, 3) uint8 array from HDF5 (per-frame video data)
        :param task_description: Robot task description
        :param per_step_info: Optional per-frame state (one entry per frame), used
            only by the ``dna_extra`` modality and injected into the description stage.
        :param return_details: If True, return a dict with keys ``progress``,
            ``objects``, ``description``, ``success_criteria``, ``failure``,
            ``feedback`` (like ``DNA._sample_once``) instead of just the progress
            array. Useful for per-sample logging.
        :return: Per-frame progress array in [0, 1], or a details dict if
            ``return_details``.
        """

        if per_step_info is not None and len(per_step_info) != len(frames_array):
            raise ValueError(
                f"per_step_info length {len(per_step_info)} != frame count {len(frames_array)}"
            )

        start_time = time.time()
        description_raw = None
        description = None
        text = None
        failure = None
        feedback = None

        n_retries = 5
        for i in range(n_retries):
            try:
                if self.modality == "dna_extra":
                    description_raw = await get_description_from_video_grounded_extra(
                        frames_array,
                        task_description,
                        model_id=self.model_name,
                        thinking_level=self.thinking_level,
                        extra_instruction=self.extra_instruction_description,
                        per_step_info=per_step_info,
                    )
                    description = {i: d["description"] for i, d in enumerate(description_raw)}
                    results = await get_progress_from_description_failure_extra(
                        description,
                        task_description,
                        model_id=self.model_name,
                        thinking_level=self.thinking_level,
                        extra_instruction=self.extra_instruction_assess,
                    )
                    payload = response_to_json(results)
                    progress = payload["progress"]
                    failure = payload.get("failure")
                    feedback = payload.get("feedback")
                    text = results.text

                    if len(progress) != len(frames_array):
                        raise ValueError(f"Progress length mismatch: {len(progress)} != {len(frames_array)}")

                    # normalize progress to [0, 1]
                    progress = np.array(progress) / 100.0
                    break

                modality, processing, reasoning = self.modality.rsplit("_", 2)
                assert modality in ["video", "video_roboreward", "video_grounded"]
                assert processing in ["hierarchy", "endtoend", "naive"]
                assert reasoning in ["single", "roboreward"]

                if processing == "naive":
                    assert modality in ["video"]
                    results = await get_progress_from_video_naive(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                    progress = response_to_json(results)["progress"]
                    text = results.text

                if processing == "endtoend":
                    assert modality in ["video", "video_roboreward"]

                    if modality == "video":
                        results = await get_progress_from_video(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                        progress = response_to_json(results)["progress"]
                        text = results.text
                    elif modality == "video_roboreward":
                        results = await get_progress_from_video_roboreward(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                        score = response_to_json(results)["score"]
                        normalized = (score - 1) / 4 * 100
                        progress = [normalized] * len(frames_array)
                        text = results.text

                if processing == "hierarchy":
                    assert modality in ["video", "video_grounded"]

                    if modality == "video":
                        description_raw = await get_description_from_video(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                    elif modality == "video_grounded":
                        description_raw = await get_description_from_video_grounded(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)

                    description = {i: d["description"] for i, d in enumerate(description_raw)}

                    if reasoning == "single":
                        results = await get_progress_from_description(description, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                        progress = response_to_json(results)["progress"]
                        text = results.text
                    elif reasoning == "roboreward":
                        results = await get_progress_from_description_roboreward(description, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                        score = response_to_json(results)["score"]
                        normalized = (score - 1) / 4 * 100
                        progress = [normalized] * len(frames_array)
                        text = results.text

                if len(progress) != len(frames_array):
                    raise ValueError(f"Progress length mismatch: {len(progress)} != {len(frames_array)}")

                # normalize progress to [0, 1]
                progress = np.array(progress) / 100.0

                break

            except Exception as e:
                if self.verbose >= 1:
                    import traceback
                    print("error computing rewards for", task_description, "\n exception: ", e)
                    traceback.print_exc()
                    print("retrying..." if i < n_retries - 1 else "max retries exceeded, returning zeros")
                if i >= n_retries - 1:
                    progress = np.zeros(len(frames_array))

        objects = get_cached_objects(frames_array, task_description)
        success_criteria = _extract_completion_state(text)

        if self.verbose >= 1:
            log_run(
                task_description,
                objects=objects,
                description=description,
                success_criteria=success_criteria,
                progress=progress,
                failure=failure,
                feedback=feedback,
            )

        # Per-sample video/text logging (flat, numbered). Skipped when the caller
        # requests details, since it does its own per-example logging instead.
        if self.verbose >= 2 and self.root_dir is not None and not return_details:
            max_progress_pct = max(progress) * 100 if len(progress) else 0.0

            video = progress_video(frames_array, progress, task_description)
            imageio.mimwrite(f"{self.root_dir}/{self.ctr}_video_{max_progress_pct:.0f}.mp4", video, fps=1)
            imageio.mimwrite(f"{self.root_dir}/{self.ctr}_video_raw_{max_progress_pct:.0f}.mp4", frames_array, fps=1)

            text_history_path = f"{self.root_dir}/{self.ctr}_text_history_{max_progress_pct:.0f}.txt"

            with open(text_history_path, "w") as f:
                f.write(task_description + "\n" + str(description_raw) + "\n" + str(description) + "\n" + str(text))
            self.ctr += 1

        if self.verbose >= 1:
            print(f"Full computation took: {time.time() - start_time} seconds")

        if return_details:
            return {
                "progress": progress,
                "objects": objects,
                "description": description,
                "success_criteria": success_criteria,
                "failure": failure,
                "feedback": feedback,
            }
        return progress

    async def compute_progress_async(self, frames_array: np.ndarray, task_description: str = "", *, n_samples: int = 1, per_step_info: Optional[list] = None, return_details: bool = False):
        """
        Async version of compute_progress with multi-sample support.

        :param frames_array: (N, H, W, 3) uint8 array from HDF5 (per-frame video data)
        :param task_description: Robot task description
        :param n_samples: Number of independent pipeline runs (concurrent); returns mean.
        :param per_step_info: Optional per-frame state (one entry per frame), used only
            by the ``dna_extra`` modality and injected into the description stage. The
            same value is passed to every sample.
        :param return_details: If True, return the list of per-sample detail dicts
            (see ``_compute_progress_once_async``) instead of the mean progress array.
        :return: (N,) mean progress array in [0, 1], or a list of per-sample dicts
            if ``return_details``.
        """
        n_samples = max(1, int(n_samples))
        results = await asyncio.gather(
            *[self._compute_progress_once_async(frames_array, task_description, per_step_info=per_step_info, return_details=return_details) for _ in range(n_samples)]
        )
        if return_details:
            return list(results)
        if n_samples == 1:
            return results[0]
        return np.mean(results, axis=0)

    def compute_progress(self, frames_array: np.ndarray, task_description: str = "", *, n_samples: int = 1, per_step_info: Optional[list] = None, return_details: bool = False):
        """
        Compute per-frame progress predictions.

        :param frames_array: (N, H, W, 3) uint8 array (per-frame video data)
        :param task_description: Robot task description
        :param n_samples: Number of independent pipeline runs (concurrent); returns mean.
        :param per_step_info: Optional per-frame state (one entry per frame), used only
            by the ``dna_extra`` modality and injected into the description stage.
        :param return_details: If True, return the list of per-sample detail dicts
            instead of the mean progress array.
        :return: (N,) mean progress array in [0, 1], or a list of per-sample dicts
            if ``return_details``.
        """
        return asyncio.run(self.compute_progress_async(frames_array, task_description, n_samples=n_samples, per_step_info=per_step_info, return_details=return_details))