import numpy as np
from dtw import dtw
from rvlm.trackers.cotracker import CoTracker
from rvlm.rewards.distance_dtw import StreamingDTW


class SimpleCoTrackerDTW:
    def __init__(self):
        # Initialize CoTracker
        self.cotracker = CoTracker(
            autocast=True,
            device="cuda",
        )
    
    def track_paths(self, video: np.ndarray, points_image_frame: dict) -> dict:
        """
        Track points through a video using CoTracker.
        
        Args:
            video: (T, H, W, 3) uint8 array
            points_image_frame: dict[name] -> (x, y) initial points in pixel coordinates
                               at frame 0 for this video
        
        Returns:
            paths_tracked: dict[name] -> (T, 2) tracked trajectories
        """
        frames = [frame for frame in video]  # Convert to list of frames
        paths_tracked = self.cotracker.track(frames, points_image_frame)
        return paths_tracked
    
    def compute_dtw_distance(
        self, 
        reference: dict, 
        query: dict,
    ) -> dict:
        """
        Compute the DTW distance between the reference and query trajectories.
        Computes DTW once per trajectory and extracts per-timestep costs
        from the pre-computed cost matrix.
        
        Args:
            reference: dict[name] -> (T, 2) reference trajectories
            query: dict[name] -> (T, 2) query trajectories
        
        Returns:
            per_path_costs: dict[name] -> (T,) DTW distance for each trajectory
        """
        keys = list(reference.keys())
        per_path_costs = {k: [] for k in keys}
        alignments = {k: [] for k in keys}

        for k in keys:
            ref_traj = reference[k]
            query_traj = query[k]

            # Compute DTW once with the full query
            alignment = dtw(
                x=ref_traj, 
                y=query_traj, 
                step_pattern='asymmetric',
                open_begin=True,
                open_end=True,
                keep_internals=True
            )

            # Extract per-prefix costs from the pre-computed cost matrix
            N = len(ref_traj)
            for t in range(1, len(query_traj) + 1):
                distance = np.nanmin(alignment.costMatrix[-1, :t]) / N
                per_path_costs[k].append(distance)
                alignments[k].append(alignment)

        return {k: np.array(v) for k, v in per_path_costs.items()}, {k: np.array(v) for k, v in alignments.items()}


    def compute_reward(self, path_reference: dict, path_query: dict, beta: float = 5e-2, return_alignments: bool = False):
        """
        Compute the reward for the query trajectory based on the reference trajectory.

        Args:
            path_reference: dict[name] -> (T, 2) reference trajectories
            path_query: dict[name] -> (T, 2) query trajectories
            beta: float, the beta parameter for the exponential function
        
        """

        per_obj_costs, alignments = self.compute_dtw_distance(path_reference, path_query)
        per_obj_rewards = {}
        for key in per_obj_costs:
            costs = per_obj_costs[key]
            baseline = costs[0]
            per_obj_rewards[key] = 1 - costs / baseline
        if return_alignments:
            return per_obj_rewards, alignments
        else:
            return per_obj_rewards