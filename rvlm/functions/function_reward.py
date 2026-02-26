import sys
sys.path.append('../reward_vlm')  # adjust relative to dsrl/

import numpy as np
from rvlm.annotator.annotator_reward import SimpleCoTrackerDTW
from rvlm.requests.helpers import (
    get_obj_labels,
    get_obj_points_from_labels,
    get_obj_paths_from_points,
    postprocess_obj_paths,
)
from rvlm.utils.conversion import gemini_to_image_frame
from rvlm.utils.visualize_batch import render_debug_video

# Global: holds CoTracker model (expensive to init), reused across all calls
_tracker: SimpleCoTrackerDTW = None
# Cache Gemini-computed reference data, keyed by language_instruction
_reference_cache: dict = {}


def _get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = SimpleCoTrackerDTW()
    return _tracker

_get_tracker()

def relabel_reward(episode_obs, episode_rewards, episode_infos, language_instruction):
    return episode_rewards
    
    tracker = _get_tracker()

    # (T, H, W, 3)
    video = np.stack([info["rgb_i"] for info in episode_infos])
    img = video[0]
    H, W = img.shape[:2]

    labels = episode_infos[0]["labels"]

    points_g = episode_infos[0]["points_g"]
    points_i = episode_infos[0]["points_i"]

    gemini = False
    if gemini and language_instruction not in _reference_cache:
        # 3) Get reference (ideal) paths from Gemini
        predicted_path_g = get_obj_paths_from_points(
            task=language_instruction, img=img, points=points_g,
            n_points=7, temperature=0.2, thinking_budget=-1,
        )
        reference_path_g = reference_path_i
        
        _reference_cache[language_instruction] = {
            "obj_points": obj_points,
            "reference_path_g": predicted_path_g,
            "reference_path_i": gemini_to_image_frame(predicted_path_g, H, W)
        }
    else:
        offline_path_i = episode_infos[0]["path_i"]
        _reference_cache[language_instruction] = {
            "reference_path_i": offline_path_i
        }

    cached = _reference_cache[language_instruction]
    
    tracked_path_i = tracker.track_paths(video, points_i)

    per_obj_rewards, per_obj_alignments = tracker.compute_reward(cached["reference_path_i"], tracked_path_i, return_alignments=True)

    per_obj_rewards_arr = np.stack(list(per_obj_rewards.values()), axis=0)  # (num_objects, T)
    mean_rewards = np.mean(per_obj_rewards_arr, axis=0)  # (T,)

    # Render debug video: (T, H, W*n_panels, 3)
    # debug_video = render_debug_video(
    #     video=video,
    #     tracked_paths=tracked_path_i,
    #     reference_paths=cached["reference_path_i"],
    #     per_obj_rewards=per_obj_rewards,
    #     per_obj_alignments=per_obj_alignments,
    #     cumulative_rewards=mean_rewards,
    # )
    debug_video = None

    return (episode_rewards + mean_rewards).tolist(), debug_video