# Observation wrapper (lightweight)
# - reset: run point prediction, select closest path, augment obs
# - step: augment obs with path state, add highres/path/points to info
# NO tracking, NO reward computation

import numpy as np
import gym
from gym import spaces

from rvlm.requests.helpers import get_obj_points_from_labels, postprocess_obj_paths

from scipy.interpolate import interp1d

from rvlm.utils.conversion import gemini_to_image_frame

def resample_by_arc_length(traj, M):
    """Resample trajectory to M points with uniform spacing along the curve."""
    N = traj.shape[0]
    diffs = np.diff(traj, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    arc_length = np.zeros(N)
    arc_length[1:] = np.cumsum(segment_lengths)

    total_length = arc_length[-1]
    if total_length == 0:
        return np.tile(traj[0], (M, 1))

    arc_length_norm = arc_length / total_length
    _, unique_idx = np.unique(arc_length_norm, return_index=True)
    unique_idx = np.sort(unique_idx)

    arc_length_unique = arc_length_norm[unique_idx]
    traj_unique = traj[unique_idx]

    kind = 'cubic' if len(unique_idx) >= 4 else 'linear'
    interpolator = interp1d(arc_length_unique, traj_unique, axis=0, kind=kind)
    uniform_arc = np.linspace(0, 1, M)
    return interpolator(uniform_arc)


def compute_path_state(img, series, labels, resample_length=-1):
    H, W = img.shape[:2]
    path_state = []
    for k in sorted(labels):
        sample = series[k]
        if resample_length > 0:
            sample = resample_by_arc_length(sample, resample_length)
        sample = sample + np.random.uniform(-5, 5, sample.shape)
        sample = np.clip(sample / np.array([H, W]), 0.0, 1.0)
        path_state.append(sample)
    return np.concatenate([p.reshape(-1) for p in path_state])


class ObsWrapper(gym.Env):
    """
    Lightweight observation wrapper.

    1) Renders highres images for VLM pointing and downstream relabeling
    2) Queries Gemini for object points (pointing)
    3) Selects the closest (or fixed) offline path
    4) Adds path_state to observations
    5) Exposes rgb_i, path, and obj_points in step info

    No tracking, no reward computation.

    Expected wrapper chain (inner -> outer):
        robomimic env -> RobomimicImageWrapper -> ObservationWrapperRobomimic -> ObsWrapper
    """

    def __init__(
        self,
        env,
        dataset_path: str,
        task: str = "Put the hammer and the screwdriver in the toolbox",
        resample_length: int = 10,
        use_path_state: bool = True,
        use_path_fix: bool = False,
        highres_hw: tuple = (256, 256),
        render_camera_name: str = "agentview",
    ):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.task = task
        self.resample_length = resample_length
        self.use_path_state = use_path_state
        self.use_path_fix = use_path_fix
        self.highres_hw = highres_hw
        self.render_camera_name = render_camera_name

        # load paths and labels from dataset
        import h5py
        from rvlm.buffers.kin_buffer import KinBuffer

        with h5py.File(dataset_path, "r", swmr=True) as f:
            for dk in f["data"].keys():
                self.labels = f["data"][dk].attrs["obj_labels"].tolist()
                break

        path_buffer = KinBuffer(dataset_path, ["path_" + l for l in self.labels])
        self.path_buffer_dict = {
            k.replace("path_", ""): v
            for k, v in path_buffer.get_dict().items()
        }

        self.path_cache = {}
        self.obj_points_processed = None

        if self.use_path_state:
            self._update_observation_space()

    def _update_observation_space(self):
        path_state_dim = 2 * self.resample_length * len(self.labels)
        self.observation_space.spaces["path_state"] = spaces.Box(
            low=-1.0, high=1.0, shape=(path_state_dim,), dtype=np.float32
        )

    def _get_highres_img(self):
        """Render highres image via the robomimic env (3 levels down the wrapper chain)."""
        h, w = self.highres_hw
        # ObsWrapper.env -> ObservationWrapperRobomimic.env -> RobomimicImageWrapper.env -> robomimic env
        return self.env.env.env.render(
            mode="rgb_array", height=h, width=w,
            camera_name=self.render_camera_name,
        )

    def _select_path(self, obj_points):
        """Select the closest offline path (or a fixed one)."""
        for k in self.path_buffer_dict:
            n_samples = len(self.path_buffer_dict[k])
            break

        init_states = []
        for i in range(n_samples):
            init_states.append(
                np.concatenate([v[i][0] for v in self.path_buffer_dict.values()], axis=0)
            )

        curr_state = np.concatenate(
            [obj_points[k] for k in self.path_buffer_dict.keys()], axis=0
        )

        dists = [np.linalg.norm(curr_state - s) for s in init_states]
        idx = np.argmin(dists)

        if self.use_path_fix:
            idx = self.use_path_fix
            print("fixing path id to", idx)

        for k in self.path_buffer_dict:
            self.path_cache[k] = [self.path_buffer_dict[k][idx]]

    def _add_path_state(self, obs, img):
        if self.use_path_state:
            obs["path_state"] = compute_path_state(
                img,
                {k: v[0] for k, v in self.path_cache.items()},
                self.labels,
                self.resample_length,
            )
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        img = self._get_highres_img()

        # # pointing: query VLM for object points
        # points_g = get_obj_points_from_labels(
        #     self.task, img, self.labels, temperature=0.0, thinking_budget=0
        # )

        # HACK: tracking dataset mismatch - hard code labels and skip pointing call
        self.labels = ["hammer_red_handle", "screwdriver_black_handle"]
        self.points_g = [
            {"point": [298, 100], "label": "hammer_red_handle"},
            {"point": [680, 398], "label": "screwdriver_black_handle"}
        ]
        # HACK: tracking dataset mismatch - hard code labels and skip pointing call

        H, W = img.shape[:2]
        self.points_i = gemini_to_image_frame(self.points_g, H, W)
        
        # select closest (or fixed) path
        self._select_path(self.points_i)

        obs = self._add_path_state(obs, img)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        img = self._get_highres_img()

        obs = self._add_path_state(obs, img)

        # highres image for downstream relabeling
        info["rgb_i"] = img

        # full selected path and VLM-predicted points
        info["labels"] = self.labels
        info["points_g"] = self.points_g
        info["points_i"] = self.points_i
        info["path_i"] = {k: v[0] for k, v in self.path_cache.items()}

        return obs, reward, done, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)
