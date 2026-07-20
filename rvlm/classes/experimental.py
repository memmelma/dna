import asyncio
import datetime
import json
import os
import time
from typing import List, Optional

import imageio
import numpy as np

from rvlm.functions.description import get_cached_objects
from rvlm.functions.terminal_log import log_run
from rvlm.functions.viz import progress_video
from rvlm.functions.video_to_progress import (
    get_description_from_all_frames,
    get_description_from_all_frames_grounded,
    get_description_from_all_frames_ungrounded,
    get_description_from_single_frames,
    get_description_from_stacked_frames,
    get_description_from_stacked_frames_grounded,
    get_description_from_video,
    get_description_from_video_grounded,
    get_progress_from_all_frames,
    get_progress_from_description,
    get_progress_from_description_distributional,
    get_progress_from_description_rubric,
    get_progress_from_description_no_completion_state,
    get_progress_from_video,
    get_progress_from_video_naive,
    get_progress_from_video_roboreward,
    get_progress_from_description_roboreward,
    get_progress_from_description_experimental,
)


def response_to_json(response) -> dict:
    response_json = json.loads(response.text)
    try:
        return response_json[0]
    except KeyError:
        return response_json
    except Exception as e:
        raise ValueError(f"Failed to parse response: {e}")


class Experimental:
    """Research kitchen-sink estimator exposing every modality / processing /
    reasoning combination. For a minimal, release-ready interface see ``DNA``.

    ``modality`` is a compound string ``{input}_{processing}_{reasoning}``.
    """

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        thinking_level: str = "MEDIUM",
        modality: str = "video_grounded_hierarchy_single",
        video_logging: bool = False,
        text_logging: bool = False,
        terminal_logging: bool = False,
        log_dir: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.thinking_level = thinking_level
        self.modality = modality
        self.video_logging = video_logging
        self.text_logging = text_logging
        self.terminal_logging = terminal_logging
        self.ctr = 0

        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.root_dir = None
        if self.video_logging or self.text_logging:
            base = log_dir or os.path.join(os.getcwd(), "rvlm_logs")
            self.root_dir = os.path.join(base, self.timestamp)
            os.makedirs(self.root_dir, exist_ok=True)

    async def compute_progress_async(self, frames_array: np.ndarray, task_description: str = "") -> List[Optional[float]]:
        """
        Async version of compute_progress. Call this inside an existing event loop
        to allow concurrent execution across many samples.
        
        :param frames_array: (N, H, W, 3) uint8 array from HDF5 (per-frame video data)
        :param task_description: Robot task description
        :return: List of cumulative sub-task counts aligned with frame indices
        """

        start_time = time.time()
        description_raw = None
        description = None
        text = None

        n_retries = 5
        for i in range(n_retries):
            try:
                modality, processing, reasoning = self.modality.rsplit("_", 2)
                assert modality in ["video", "video_roboreward", "image", "all_frames", "all_frames_grounded", "all_frames_ungrounded", "stacked_frames", "single_frames", "stacked_frames_grounded", "video_grounded"]
                assert processing in ["hierarchy", "endtoend", "naive"]
                assert reasoning in ["single", "distributional", "rubric", "no_completion_state", "roboreward", "experimental"]

                if processing == "naive":
                    assert modality in ["video"]
                    results = await get_progress_from_video_naive(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                    progress = response_to_json(results)["progress"]
                    text = results.text

                if processing == "endtoend":
                    assert reasoning in ["single", "roboreward"]
                    assert modality in ["video", "video_roboreward", "all_frames"]

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
                    elif modality == "all_frames":
                        results = await get_progress_from_all_frames(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                        progress = response_to_json(results)["progress"]
                        text = results.text

                if processing == "hierarchy":
                    assert reasoning in ["single", "distributional", "rubric", "no_completion_state", "roboreward", "experimental"]
                    assert modality in ["video", "image", "all_frames", "stacked_frames", "single_frames", "all_frames_grounded", "all_frames_ungrounded", "stacked_frames_grounded", "video_grounded"]

                    if modality == "video":
                        description_raw = await get_description_from_video(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                    elif modality == "video_grounded":
                        description_raw = await get_description_from_video_grounded(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                    elif modality == "image":
                        description_raw = await get_description_from_all_frames(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                    elif modality == "all_frames":
                        description_raw = await get_description_from_all_frames(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                    elif modality == "all_frames_grounded":
                        description_raw = await get_description_from_all_frames_grounded(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                    elif modality == "all_frames_ungrounded":
                        description_raw = await get_description_from_all_frames_ungrounded(frames_array, model_id=self.model_name, thinking_level=self.thinking_level)
                    elif modality == "stacked_frames":
                        description_raw = await get_description_from_stacked_frames(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                    elif modality == "stacked_frames_grounded":
                        description_raw = await get_description_from_stacked_frames_grounded(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                    elif modality == "single_frames":
                        description_raw = await get_description_from_single_frames(frames_array, task_description, model_id=self.model_name, thinking_level=self.thinking_level)

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
                    elif reasoning == "no_completion_state":
                        results = await get_progress_from_description_no_completion_state(description, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                        progress = response_to_json(results)["progress"]
                        text = results.text
                    elif reasoning == "rubric":
                        results = await get_progress_from_description_rubric(description, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                        progress = response_to_json(results)["progress"]
                        text = results.text
                    elif reasoning == "experimental":
                        results = await get_progress_from_description_experimental(description, task_description, model_id=self.model_name, thinking_level=self.thinking_level)
                        progress = response_to_json(results)["progress"]
                        text = results.text
                    elif reasoning == "distributional":
                        results = await get_progress_from_description_distributional(description, task_description, model_id=self.model_name, thinking_level=self.thinking_level, k_requests=5)
                        progress = [json.loads(r.text)[0]["progress"] for r in results]
                        progress = np.mean(progress, axis=0)
                        text = json.dumps([r.text for r in results])

                if len(progress) != len(frames_array):
                    raise ValueError(f"Progress length mismatch: {len(progress)} != {len(frames_array)}")

                # normalize progress to [0, 1]
                progress = np.array(progress) / 100.0

                break

            except Exception as e:
                import traceback
                print("error computing rewards for", task_description, "\n exception: ", e)
                traceback.print_exc()
                if i < n_retries - 1:
                    print("retrying...")
                else:
                    print("max retries exceeded, returning zeros")
                    progress = np.zeros(len(frames_array))

        if self.terminal_logging:
            objects = get_cached_objects(frames_array, task_description)
            log_run(task_description, objects=objects, description=description, progress=progress)

        if (self.video_logging or self.text_logging) and self.root_dir is not None:
            max_progress_pct = max(progress) * 100 if len(progress) else 0.0

            if self.video_logging:
                video = progress_video(frames_array, progress, task_description)
                imageio.mimwrite(f"{self.root_dir}/{self.ctr}_video_{max_progress_pct:.0f}.mp4", video, fps=1)
                imageio.mimwrite(f"{self.root_dir}/{self.ctr}_video_raw_{max_progress_pct:.0f}.mp4", frames_array, fps=1)

            text_history_path = f"{self.root_dir}/{self.ctr}_text_history_{max_progress_pct:.0f}.txt"

            with open(text_history_path, "w") as f:
                f.write(task_description + "\n" + str(description_raw) + "\n" + str(description) + "\n" + str(text))
            self.ctr += 1

        print(f"Full computation took: {time.time() - start_time} seconds")

        return progress

    def compute_progress(self, frames_array: np.ndarray, task_description: str = "") -> List[Optional[float]]:
        """
        Compute per-frame progress predictions.

        :param frames_array: (N, H, W, 3) uint8 array (per-frame video data)
        :param task_description: Robot task description
        :return: (N,) array of per-frame progress values in [0, 1]
        """
        return asyncio.run(self.compute_progress_async(frames_array, task_description))