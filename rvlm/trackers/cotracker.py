import torch
import numpy as np
from typing import Optional, Dict, List
from contextlib import nullcontext


class CoTracker:
    """Offline (batch) CoTracker wrapper — processes the entire trajectory at once."""

    def __init__(self, autocast: bool = True, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_offline"
        ).to(self.device)
        self.model.eval()
        self.autocast = autocast

    def track(
        self,
        frames: List[np.ndarray],
        initial_point_dict: Dict[str, tuple],
    ) -> Dict[str, np.ndarray]:
        """
        Track named points through an entire video.

        Args:
            frames: List of (H, W, 3) uint8 numpy arrays — the full trajectory.
            initial_point_dict: {"name": (x, y), ...} in pixel coordinates
                                (points are assumed to originate at frame 0).

        Returns:
            Dict mapping point names to (T, 2) numpy arrays of (x, y) positions
            for every frame in the input.
        """
        point_keys = list(initial_point_dict.keys())
        xs = [initial_point_dict[k][0] for k in point_keys]
        ys = [initial_point_dict[k][1] for k in point_keys]
        t0 = np.zeros(len(point_keys), dtype=np.float32)

        # queries shape: (1, N_points, 3)  where 3 = (t, x, y)
        queries = torch.tensor(
            np.stack([t0, xs, ys], axis=1),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # Build video tensor: (1, T, 3, H, W)
        video = (
            torch.tensor(np.stack(frames), device=self.device)
            .float()
            .permute(0, 3, 1, 2)[None]
        )

        H, W = frames[0].shape[:2]

        with torch.inference_mode():
            with (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.autocast
                else nullcontext()
            ):
                # Single forward pass over the whole video
                pred_tracks, pred_visibility = self.model(
                    video, queries=queries
                )
                # pred_tracks:     (1, T, N, 2)  — (x, y) per frame per point
                # pred_visibility:  (1, T, N, 1)

        # (T, N, 2)
        tracks = pred_tracks[0].cpu().numpy()

        # Clamp to valid pixel bounds
        tracks[:, :, 0] = np.clip(tracks[:, :, 0], 0, W - 1)
        tracks[:, :, 1] = np.clip(tracks[:, :, 1], 0, H - 1)

        # Build the same dict interface: {name: (T, 2)}
        tracks_dict = {k: tracks[:, i, :] for i, k in enumerate(point_keys)}
        return tracks_dict