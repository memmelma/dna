import numpy as np

from rvlm.trackers.cotracker import CoTracker
from rvlm.rewards.distance_dtw import StreamingDTW


class VideoDTWAnnotator:
    def __init__(
        self,
        cotracker_device: str = "cuda",
        cotracker_autocast: bool = True,
        reward_beta: float = 5e-2,
        reward_scale: float = 1.0,
        timestep_penalty: float = -1.0,
        success_reward: float = 0.0,
        dtw_flexible_end: bool = False,
        dtw_normalize_cost: bool = True,
        dtw_alpha: float = 0.0,
        dtw_beta: float = 100.0,
    ):
        # CoTracker config
        self.cotracker_device = cotracker_device
        self.cotracker_autocast = cotracker_autocast
        self.cotracker = CoTracker(
            autocast=self.cotracker_autocast,
            device=self.cotracker_device,
        )

        # Reward config
        self.reward_beta = reward_beta
        self.reward_scale = reward_scale
        self.timestep_penalty = timestep_penalty
        self.success_reward = success_reward

        # DTW config
        self.dtw_kwargs = dict(
            flexible_end=dtw_flexible_end,
            normalize_cost=dtw_normalize_cost,
            alpha=dtw_alpha,
            beta=dtw_beta,
        )

    def forward(
        self,
        video: np.ndarray,
        language_instruction: str,
        obj_points: dict,
        reference_paths: dict,
    ):
        """
        Track points in a video with CoTracker and compute DTW-based rewards
        to given reference/GT series.

        Args:
            video: (T, H, W, 3) uint8 array.
            language_instruction: Not used in DTW, kept for API symmetry / logging.
            obj_points: dict[name] -> (x0, y0) initial points in pixel coordinates
                        at frame 0 for this video (CoTracker queries).
            series: dict[name] -> (T_ref, 2) reference / GT trajectory for DTW.

        Returns:
            paths_tracked: dict[name] -> (T, 2) tracked trajectories for this video.
            rewards: (T,) numpy array of scalar rewards over time.
        """
        T = video.shape[0]
        obj_keys = list(obj_points.keys())

        # --- 1) Track video with CoTracker from obj_points ---
        frames = [frame for frame in video]  # list of (H, W, 3)
        paths_tracked = self.cotracker.track(frames, obj_points)
        # paths_tracked: {name: (T, 2)}

        # --- 2) For each object, run DTW between reference series and tracked path ---
        per_obj_costs = {}
        for key in obj_keys:
            traj_tracked = paths_tracked[key]       # (T, 2)
            ref_series = reference_paths[key]                # (T_ref, 2) or similar

            # DTW expects series as list of (M, D) arrays; here one series: (T_ref, 2)
            dtw = StreamingDTW(series=[ref_series], **self.dtw_kwargs)

            # Non‑streaming forward over entire tracked trajectory
            # query: (T, 2) -> returns (M, T) costs (M=1 here)
            costs = dtw.forward(traj_tracked)       # (1, T)
            per_obj_costs[key] = costs[0]          # (T,)

        # --- 3) Convert distances to rewards per timestep ---
        rewards = []
        for t in range(T):
            tracking_rewards = []
            for key in obj_keys:
                dist_t = per_obj_costs[key][t]
                rew_t = np.exp(-self.reward_beta * dist_t)
                tracking_rewards.append(rew_t)

            reward_t = self.timestep_penalty + self.reward_scale * np.mean(
                tracking_rewards
            )
            rewards.append(reward_t)

        # Final timestep shaping
        rewards[-1] += self.success_reward - self.timestep_penalty

        rewards = np.array(rewards, dtype=np.float32)
        return paths_tracked, rewards