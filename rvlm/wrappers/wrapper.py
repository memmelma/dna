# Reward wrapper
# - reset: run point prediction, reset cotracker, reset reward
# - step: run tracker + compute reward

# - should work for sim and real
#     - sim: retrieval from precomputed VLM dataset (generate adhoc from first observations)
#     - real: run before every rollout (or once if poses don't change much?)

# Experiments: don't use state anymore! -> dilutes results and procrastinates real implementation 

import os
import time
import json
import datetime
import numpy as np
import gymnasium as gym

from rvlm.requests.prompts import get_alternative_pointing_prompt
from rvlm.requests.gemini_utils import img_to_mime, create_config, call_gemini_robotics_er
from rvlm.requests.helpers import get_obj_points_from_labels, postprocess_obj_paths

def _plot_tracking_debug(img, paths, tracking_series, tracking_rewards, save_path=None):
    """
    Plot tracked paths against reference series for debugging.
    
    Args:
        img: The current image frame (H, W, C)
        paths: Dict of tracked paths from cotracker {label: (T, 2)}
        tracking_series: Dict of best-matching reference series {label: (T, 2)}
        tracking_rewards: Dict of rewards per object {label: float}
    """
    import matplotlib.pyplot as plt
    
    obj_labels = tracking_series.keys()

    n_plots = len(obj_labels)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    
    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]
    
    for ax, k in zip(axes, obj_labels):
        # Show image background
        ax.imshow(img)
        
        # Plot tracked path (from cotracker)
        tracked = paths[k].astype(int)
        ax.plot(tracked[:, 0], tracked[:, 1], 'tab:blue', linewidth=3, label='Tracked')
        
        # Plot reference series (best match)
        ref = tracking_series[k].astype(int)
        ax.plot(ref[:, 0], ref[:, 1], 'tab:orange', linewidth=3, linestyle="--", label='Reference')
        
        # Mark start points
        ax.scatter(tracked[0, 0], tracked[0, 1], c='tab:blue', marker='x', s=250, linewidth=3, zorder=5)
        ax.scatter(ref[0, 0], ref[0, 1], c='tab:orange', marker='x', s=100, linewidth=3, zorder=5)
        
        ax.set_title(f'Point: {k}\nTimestep: {len(tracked)}\nReward: {tracking_rewards[k]:.4f}')
        ax.legend(loc='upper right')
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

class TrackingRewardWrapper(gym.Wrapper):
    """
    Wrapper to add tracking rewards to an environment.

    1) Queries Gemini for object points
    2) Tracks object points in the image observations
    3) Computes distance between tracked points and offline tracks
    4) Computes reward based on distance

    Args:
        env: The environment to wrap
        img_key: The key of the image observation
        obj_labels: The keys of the object points to track
        
        offline_paths: The offline tracks to compare to
        
        reward_type: The type of distance metric to compute
        reward_normalize: Whether to normalize the reward by timesteps
        reward_beta: The beta parameter for the exponential reward
        reward_scale: The scale parameter for the reward
    """

    def __init__(
        self, env: gym.Env,

        # TODO: remove default values
        img_key: str,
        obj_labels: list[str],

        offline_paths: dict[str, list[np.ndarray]] = None,

        reward_type: str = "dtw",
        reward_normalize: bool = True,
        reward_beta: float = 1.0,
        reward_scale: float = 1.0,
    ):
        super().__init__(env)

        self.img_key = img_key
        
        # TODO implement tracking key prediction using gemini (outside)
        
        self.obj_labels = obj_labels
        self.obj_points = None

        self.reward_type = reward_type
        self.reward_beta = reward_beta
        self.reward_scale = reward_scale

        # TODO implement language instruction as env variable
        self.task = "Put the hammer and the screwdriver in the toolbox" # TASK_TO_LANG[env_name]

        # cotracker
        from rvlm.trackers.streaming_cotracker import StreamingCoTracker
        self.cotracker = StreamingCoTracker(efficient_mode=True, device="cuda")

        # distance
        self.tracking_caches = {}
        for k in self.obj_labels:

            series = offline_paths[k]

            # initialize cached tracking metric
            if self.reward_type == "signature":
                from rvlm.rewards.distance_signature import StreamingSignature

                signature_depth = 3
                self.tracking_caches[k] = StreamingSignature(
                    series=series, m=signature_depth, normalize_cost=reward_normalize, add_time_obs=True
                )
            elif self.reward_type == "dtw":
                from rvlm.rewards.distance_dtw import StreamingDTW

                self.tracking_caches[k] = StreamingDTW(
                    series=series, flexible_end=True, normalize_cost=reward_normalize
                )

    def reset(self, **kwargs):

        self._unique_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        obs, info = self.env.reset(**kwargs)

        img = obs[self.img_key]

        # query VLM for points (every time) -> works decent w/o reasoning
        obj_points = get_obj_points_from_labels(self.task, img, self.obj_labels, temperature=0.2, thinking_budget=0)

        # convert gemini format to dict
        H, W = img.shape[:2]
        obj_points_processed = postprocess_obj_paths(obj_points, H, W)

        # reset cotracker
        self.cotracker.reset(obj_points_processed)

        # reset distance
        for k in self.tracking_caches.keys():
            self.tracking_caches[k].reset()

        return obs, info

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        img = obs[self.img_key].copy()

        # run tracker - returns 0:t
        paths = self.cotracker.step(img)
       
        # compute distance
        tracking_rewards = {}
        tracking_series = {}
        for k in self.obj_labels:

            # TODO does this work for DTW? -> sanity check but I think yes!
            dists = self.tracking_caches[k].step(paths[k][-1:])
            dists = np.array(dists)

            rewards = np.exp(-self.reward_beta * dists)
            
            max_idx = np.argmax(rewards)
            tracking_rewards[k] = rewards[max_idx]
            tracking_series[k] = self.tracking_caches[k].series[max_idx]

        # plot if enough points and update every 32 steps
        min_series_length = min([len(s) for s in self.tracking_caches[k].series])
        if len(paths[k]) > min_series_length * 0.8 and len(paths[k]) % 32 == 0:
            save_dir = "tracking"
            os.makedirs(save_dir, exist_ok=True)
            _plot_tracking_debug(img, paths, tracking_series, tracking_rewards, save_path=os.path.join(save_dir, f"{self._unique_id}.png"))

        # compute reward
        reward = reward + self.reward_scale * np.sum(list(tracking_rewards.values()))

        return obs, reward, terminated, truncated, info