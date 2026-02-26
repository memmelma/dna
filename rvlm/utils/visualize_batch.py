"""
Batched trajectory visualization for reward relabeling.

Produces a debug video (T, H_out, W_out, 3) by calling the same plotting
functions used in TrackingRewardWrapper (visualize.py), but over the full
episode at once.

Per frame:
  [tracking_debug | alignment_obj_1 | ... | alignment_obj_N | reward_curves]
"""

import numpy as np
from rvlm.utils.visualize import plot_tracking_debug, plot_alignment, plot_costs


def render_debug_video(
    video: np.ndarray,
    tracked_paths: dict,
    reference_paths: dict,
    per_obj_rewards: dict,
    per_obj_alignments: dict,
    cumulative_rewards: np.ndarray,
    colors=None,
) -> np.ndarray:
    """
    Render a full debug video for one episode.

    Args:
        video:              (T, H, W, 3) uint8 episode frames.
        tracked_paths:      dict[label] -> (T, 2) tracked pixel trajectories.
        reference_paths:    dict[label] -> (N, 2) reference pixel trajectories.
        per_obj_rewards:    dict[label] -> list of length T, per-object per-step rewards.
        per_obj_alignments: optional dict[label] -> list of dtw alignment objects.
        cumulative_rewards: (T,) combined reward at each step.
        colors:             optional list of colors for each object.

    Returns:
        debug_video: (T, H, W * n_panels, 3) uint8 array.
    """
    T = video.shape[0]
    # obj_keys = sorted(tracked_paths.keys())

    frames = []
    for t in range(T):
        curr_frame = []

        curr_frame.append(plot_costs([{k:v[i] for i, (k,v) in enumerate(per_obj_rewards.items())}][:t+1]))

        for k in per_obj_alignments.keys():
            curr_frame.append(plot_alignment(per_obj_alignments[k][t]))

        curr_frame.append(plot_tracking_debug(
            img=video[t],
            paths=tracked_paths,
            tracking_series=reference_paths,
            tracking_rewards={k:v[t] for k,v in per_obj_rewards.items()},
            rewards=cumulative_rewards[:t+1],
            colors=colors,
        ))
        frames.append(curr_frame)

    return np.stack([np.concatenate(frame, axis=1) for frame in frames])

    # Build per-step tracking_rewards dicts (same format as wrapper uses)
    tracking_rewards_list = []
    for t in range(T):
        tracking_rewards_list.append({
            k: per_obj_rewards[k][t] for k in obj_keys
        })

    # Build cumulative reward list (same format as wrapper.self.rewards)
    rewards_list = cumulative_rewards.tolist()

    alignments_panels = {}
    if per_obj_alignments is not None:
        for k in obj_keys:
            if k in per_obj_alignments:
                alignments_panels[k] = []
                for alignment in per_obj_alignments[k]:
                    alignments_panels[k].append(plot_alignment(alignment))

    # Per-frame tracking overlay
    frames = []
    for t in range(T):
        tracking_series = {k: reference_paths[k] for k in obj_keys}
        paths = {k: tracked_paths[k][:t + 1] for k in obj_keys}
        tracking_rewards = tracking_rewards_list[t]
        rewards_so_far = rewards_list[:t + 1]

        tracking_panel = plot_tracking_debug(
            img=video[t],
            paths=paths,
            tracking_series=tracking_series,
            tracking_rewards=tracking_rewards,
            rewards=rewards_so_far,
            colors=colors,
        )

        panels = [tracking_panel] + [alignments_panels[k] for k in obj_keys]
        frame = np.concatenate(panels, axis=1)
        frames.append(frame)

    return np.stack(frames)
