# Reward wrapper
# - reset: run point prediction, reset cotracker, reset reward
# - step: run tracker + compute reward

# - should work for sim and real
#     - sim: retrieval from precomputed VLM dataset (generate adhoc from first observations)
#     - real: run before every rollout (or once if poses don't change much?)

# Experiments: don't use state anymore! -> dilutes results and procrastinates real implementation 

import os
import time
import pickle
import datetime
import numpy as np
import gymnasium as gym

from rvlm.requests.prompts import get_alternative_pointing_prompt
from rvlm.requests.gemini_utils import img_to_mime, create_config, call_gemini_robotics_er
from rvlm.requests.helpers import get_obj_points_from_labels, postprocess_obj_paths

from scipy.interpolate import interp1d


def resample_by_arc_length(traj, M):
    """
    Resample trajectory to M points with uniform spacing along the curve.
    traj: (N, 3) array
    M: target number of points
    """
    N = traj.shape[0]
    
    # Compute distances between consecutive points
    diffs = np.diff(traj, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    
    # Cumulative arc length
    arc_length = np.zeros(N)
    arc_length[1:] = np.cumsum(segment_lengths)
    
    total_length = arc_length[-1]
    if total_length == 0:
        return np.tile(traj[0], (M, 1))
    
    # Normalize to [0, 1]
    arc_length_norm = arc_length / total_length
    
    # Remove duplicate arc-length values (keep first occurrence)
    _, unique_idx = np.unique(arc_length_norm, return_index=True)
    unique_idx = np.sort(unique_idx)  # preserve order
    
    arc_length_unique = arc_length_norm[unique_idx]
    traj_unique = traj[unique_idx]
    
    # Need at least 4 points for cubic, fall back to linear if fewer
    kind = 'cubic' if len(unique_idx) >= 4 else 'linear'
    
    # Create interpolator
    interpolator = interp1d(arc_length_unique, traj_unique, axis=0, kind=kind)
    
    # Sample at uniform arc-length intervals
    uniform_arc = np.linspace(0, 1, M)
    resampled = interpolator(uniform_arc)
    
    return resampled

def compute_path_state(img, series, obj_labels, resample_length=-1):

    H, W = img.shape[:2]

    path_state = []
    # make sure order is consistent
    for k in sorted(obj_labels):
        sample = series[k]
        # resample
        if resample_length > 0:
            sample = resample_by_arc_length(sample, resample_length)
        # +/- 5 pixels noise
        sample = sample + np.random.uniform(-5, 5, sample.shape)
        # normalize to [0, 1]
        sample = np.clip(sample / np.array([H, W]), 0.0, 1.0)

        path_state.append(sample)

    # flatten to len(obj_labels) x resample_length x 2
    return np.concatenate([p.reshape(-1) for p in path_state])

def compute_path_img(img, series, obj_labels, resample_length=-1, save=None):

    from tool_use.envs.utils import add_path_2d_to_img, COLORS

    colors = [(1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (0.0, 1.0, 0.5), (1.0, 0.0, 0.5)]
    tmp_img = img.copy()
    for i,k in enumerate(sorted(obj_labels)):
        sample = series[k]
        if resample_length > 0:
            sample = resample_by_arc_length(sample, resample_length)
        tmp_img = add_path_2d_to_img(tmp_img, sample, color=colors[i], line_size=2)

    if save:
        import matplotlib.pyplot as plt
        plt.imsave(save, tmp_img)
    
    return tmp_img

def _plot_tracking_debug(img, paths, tracking_series, tracking_rewards, save_dir=None, unique_id=None):
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
    
    # save figure
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f"{unique_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # dump data
    data = {
        "img": img,
        "paths": paths,
        "tracking_series": tracking_series,
        "tracking_rewards": tracking_rewards,
    }
    pickle.dump(data, open(os.path.join(save_dir, f"{unique_id}.pkl"), "wb"))

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
        # obj_labels: list[str],
        # offline_paths: dict[str, list[np.ndarray]] = None,
        dataset_path: str = None,

        reward_type: str = "dtw",
        reward_normalize: bool = True,
        reward_beta: float = 1e-2,
        reward_scale: float = 1.0,

        resample_length: int = -1,
        use_path_img: bool = False,
        use_path_state: bool = False,
    ):
        super().__init__(env)

        self.evaluate = env.unwrapped.evaluate
        self.first_reset = True

        # EXPERIMENTAL

        if dataset_path:

            from rvlm.buffers.kin_buffer import KinBuffer

            # NOTE: new way of loading paths from hdf5
            import h5py
            with h5py.File(dataset_path, "r", swmr=True) as f:
                for dk in f["data"].keys():
                    obj_labels = f["data"][dk].attrs["obj_labels"].tolist()
                    keys = list(f["data"][dk]["obs"].keys())
                    break

            path_buffer = KinBuffer(dataset_path, ["path_" + l for l in obj_labels])

            self.path_buffer_dict = {}
            # HACK: remove "path_" from keys
            for k,v in path_buffer.get_dict().items():
                self.path_buffer_dict[k.replace("path_", "")] = v

        assert resample_length == -1, "Resampling breaks streaming DTW"
        # EXPERIMENTAL


        self.img_key = img_key
        
        # TODO implement tracking key prediction using gemini (outside)
        self.obj_labels = obj_labels
        self.obj_points = None

        self.reward_type = reward_type
        self.reward_normalize = reward_normalize
        self.reward_beta = reward_beta
        self.reward_scale = reward_scale

        # TODO implement language instruction as env variable
        self.task = "Put the hammer and the screwdriver in the toolbox" # TASK_TO_LANG[env_name]

        # cotracker
        from rvlm.trackers.streaming_cotracker import StreamingCoTracker
        self.cotracker = StreamingCoTracker(efficient_mode=True, device="cuda")

        # TODO pick path based on closest initial condition
        self.path_cache = {}
        self.obj_points_processed = None
        self.resample_length = resample_length

        self.use_path_img = use_path_img
        self.use_path_state = use_path_state

        self._update_observation_space()

    def _update_observation_space(self):
        """
        Update the observation space to include path obs for each label (separate state tags?)
        """
        if self.use_path_state:
            path_state_dim = 2 * self.resample_length * len(self.obj_labels)
            self.env.observation_space.spaces["path_state"] = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(path_state_dim,), dtype=np.float32
            )
        # if self.use_path_img:
        #     paths_img_dim = self.env.observation_space.spaces[self.img_key].shape
        #     self.env.observation_space.spaces["paths_img"] = gym.spaces.Box(
        #         low=0, high=255, shape=paths_img_dim, dtype=np.uint8
        #     )

    def reset_tracking(self):

        # distance
        self.tracking_caches = {}
        for k in self.obj_labels:

            series = self.path_cache[k]

            # initialize cached tracking metric
            if self.reward_type == "signature":
                from rvlm.rewards.distance_signature import StreamingSignature

                signature_depth = 3
                self.tracking_caches[k] = StreamingSignature(
                    series=series, m=signature_depth, normalize_cost=self.reward_normalize, add_time_obs=True
                )
            elif self.reward_type == "dtw":
                from rvlm.rewards.distance_dtw import StreamingDTW

                # alpha=0.75, beta=100 allows for up to 25% slowdown and 100x speedup of the tracked trajectory compared to the reference trajectory -> required when using VLM points since len(vlm) << len(tracker)
                self.tracking_caches[k] = StreamingDTW(
                    series=series, flexible_end=False, normalize_cost=self.reward_normalize, alpha=0.0, beta=100.0
                )
    
    def init_closests_path(self, obj_points):

        
        # if choose closest path
        if True:

            for k in self.path_buffer_dict.keys():
                n_samples = len(self.path_buffer_dict[k])
                break

            # compute "features" / initial states for each sample
            init_states = []
            for i in range(n_samples):
                init_states.append(np.concatenate([v[i][0] for v in self.path_buffer_dict.values()], axis=0))

            # compute current state
            curr_state = np.concatenate([obj_points[k] for k in self.path_buffer_dict.keys()], axis=0)

            # compute distances
            dists = [np.linalg.norm(curr_state - s) for s in init_states]
            idx = np.argmin(dists)

            # update path cache
            for k in self.path_buffer_dict.keys():
                self.path_cache[k] = [self.path_buffer_dict[k][idx]]

        else:
            pass
            # query VLM
            # self.path_cache = self.path_buffer_dict

    def reset(self, **kwargs):

        self._unique_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        obs, info = self.env.reset(**kwargs)

        img = obs[self.img_key]

        if self.first_reset or not self.evaluate:
            # query VLM for points (every time) -> works decent w/o reasoning
            obj_points = get_obj_points_from_labels(self.task, img, self.obj_labels, temperature=0.2, thinking_budget=0)

            # convert gemini format to dict
            H, W = img.shape[:2]
            self.obj_points_processed = postprocess_obj_paths(obj_points, H, W)


            # EXPERIMENTAL
            self.init_closests_path(self.obj_points_processed)
            self.reset_tracking()

            self.first_reset = False

        # reset cotracker
        self.cotracker.reset(self.obj_points_processed)

        # reset distance
        for k in self.tracking_caches.keys():
            self.tracking_caches[k].reset()

        if self.use_path_state:
            obs["path_state"] = compute_path_state(img, {k:v[0] for k,v in self.path_cache.items()}, self.obj_labels, self.resample_length)
        if self.use_path_img:
            obs[self.img_key] = compute_path_img(img, {k:v[0] for k,v in self.path_cache.items()}, self.obj_labels, self.resample_length, save=f"path_img_{self._unique_id}.png")
        
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

            assert len(paths[k].shape) == 2, f"Paths for {k} have shape {paths[k].shape}"

            # TODO: if len(paths) > len(series), resample for DTW?
            dists = self.tracking_caches[k].step(paths[k][-1:])
            dists = np.array(dists)
            
            assert len(dists) == 1, f"Dists for {k} have shape {dists.shape}"

            rewards = np.exp(-self.reward_beta * dists)
            # print("mean dists", np.around(np.mean(dists), 3), "mean rewards", np.around(np.mean(rewards), 3))
            
            # max_idx = np.argmax(rewards)
            # tracking_rewards[k] = rewards[max_idx]
            
            max_idx = 0 # only one trajectory
            tracking_rewards[k] = rewards[max_idx]
            tracking_series[k] = self.tracking_caches[k].series[max_idx]

            assert len(tracking_series[k].shape) == 2, f"Tracking series for {k} have shape {tracking_series[k].shape}"


        # compute reward
        reward = reward + self.reward_scale * np.mean(list(tracking_rewards.values()))

        if self.use_path_state:
            obs["path_state"] = compute_path_state(img, {k:v[0] for k,v in self.path_cache.items()}, self.obj_labels, self.resample_length)
        if self.use_path_img:
            obs[self.img_key] = compute_path_img(img, {k:v[0] for k,v in self.path_cache.items()}, self.obj_labels, self.resample_length)

        # Add tracking_rewards to info for logging
        if info is None:
            info = {}
        info['tracking_rewards'] = tracking_rewards

        return obs, reward, terminated, truncated, info