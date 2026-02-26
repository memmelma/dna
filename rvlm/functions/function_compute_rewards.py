import datetime
import asyncio
import imageio
import numpy as np
import os
import pickle
from rvlm.functions.function_tracking import video_to_dense_reward
from rvlm.functions.function_rubric import compute_language_based_subtasks, rubric_to_dense_reward
from rvlm.utils.visualize_np import plot_full_paths_on_first_frame, plot_rewards

async def compute_rewards(video: np.ndarray, language_instruction: str):
    """
    Custom reward function that scores each frame of a video trajectory.

    Args:
        video: numpy array of images with shape (T, H, W, C)
        language_instruction: task description string

    Returns:
        rewards: numpy array of rewards with shape (T,), one per frame
        success_probs: numpy array of success probabilities with shape (T,)
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results = await asyncio.gather(
        video_to_dense_reward(video, language_instruction),
        compute_language_based_subtasks(language_instruction, video)
    )

    tracked_path_i, predicted_path_i, per_obj_rewards, per_obj_alignments, tracking_rewards = results[0]
    subtasks = results[1]
    subtask_rewards = rubric_to_dense_reward(subtasks, len(video), delay=0)

    # combine rewards
    rewards = tracking_rewards + subtask_rewards

    # Success probability = fraction of completed subtasks
    if isinstance(subtasks, list) and len(subtasks) > 0:
        n_success = sum(1 for task in subtasks if task.get("success", False))
        success_prob = n_success / len(subtasks)
    else:
        n_success, success_prob = 0, 0.0
    success_probs = np.full(len(video), success_prob, dtype=np.float32)

    root_dir = f"reward_relabel/{language_instruction.replace(' ', '_')}/{timestamp}"
    os.makedirs(root_dir, exist_ok=True)

    # save data
    data = {
        "tracked_path_i": tracked_path_i,
        "predicted_path_i": predicted_path_i,
        "per_obj_rewards": per_obj_rewards,
        "per_obj_alignments": per_obj_alignments,
        "tracking_rewards": tracking_rewards,
        "subtask_rewards": subtask_rewards,
        "rewards": rewards,
        "success_probs": success_probs,
        "language_instruction": language_instruction,
        "video": video,
    }
    with open(f"{root_dir}/data.pkl", "wb") as f:
        pickle.dump(data, f)

    # plot rewards
    plt_tracked = plot_full_paths_on_first_frame(video=video, language_instruction=language_instruction, paths_out=tracked_path_i)
    plt_predicted = plot_full_paths_on_first_frame(video=video, language_instruction=language_instruction, paths_out=predicted_path_i)

    plt_rewards = plot_rewards(tracking_rewards, subtask_rewards)

    imageio.imwrite(f"{root_dir}/tracked.png", plt_tracked)
    imageio.imwrite(f"{root_dir}/predicted.png", plt_predicted)
    imageio.imwrite(f"{root_dir}/rewards.png", plt_rewards)

    imageio.mimwrite(f"{root_dir}/video.mp4", video, fps=1)

    with open(f"{root_dir}/subtasks.txt", "w") as f:
        f.write(str(subtasks))
    return rewards, success_probs