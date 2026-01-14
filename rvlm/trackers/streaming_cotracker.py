import torch
import numpy as np
from typing import Optional, Dict
from contextlib import nullcontext

class StreamingCoTracker:
    """Minimal streaming CoTracker wrapper."""
    
    def __init__(self, efficient_mode: bool = True, autocast: bool = True, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(self.device)
        self.model.eval()

        self.autocast = autocast
        self.efficient_mode = efficient_mode
        
    def reset(self, initial_point_dict: Dict[str, tuple]):
        """
        Initialize with named points to track.
        
        Args:
            initial_point_dict: {"name": (x, y), ...} in pixel coordinates
        """
        self.point_keys = list(initial_point_dict.keys())
        xs = [initial_point_dict[k][0] for k in self.point_keys]
        ys = [initial_point_dict[k][1] for k in self.point_keys]
        t0 = np.zeros(len(self.point_keys), dtype=np.float32)
        
        # queries shape: (1, N_points, 3) where 3 = (t, x, y)
        self.queries = torch.tensor(
            np.stack([t0, xs, ys], axis=1),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        self.frame_buffer = []
        self.is_first_step = True
        self.pred_tracks = None
        self.pred_tracks_buffer = None
        
    def _step(self, next_image: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Process next frame.
        
        Args:
            next_image: (H, W, 3) uint8 numpy array
            
        Returns:
            Dict mapping point names to (x, y) arrays, or None if warming up
        """
        self.frame_buffer.append(next_image)
        
        # Only process after we have at least step frames, and every step frames after that
        if len(self.frame_buffer) >= self.model.step and len(self.frame_buffer) % self.model.step == 0:
            # Pad buffer if less than window size
            if len(self.frame_buffer) < self.model.step * 2:
                pad_count = self.model.step * 2 - len(self.frame_buffer)
                window = [self.frame_buffer[0]] * pad_count + self.frame_buffer
            else:
                window = self.frame_buffer[-self.model.step * 2:]
            
            video_chunk = (
                torch.tensor(np.stack(window), device=self.device)
                .float()
                .permute(0, 3, 1, 2)[None]
            )  # (1, T, 3, H, W)
            
            # Max speed and reduce memory usage
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16) if self.autocast else nullcontext():
                    if self.is_first_step:
                        self.model(video_chunk, is_first_step=True, queries=self.queries)
                        self.pred_tracks, _ = self.model(video_chunk, is_first_step=False)
                        self.is_first_step = False
                    else:
                        self.pred_tracks, _ = self.model(video_chunk, is_first_step=False)
        
        if self.pred_tracks is None:
            return None
        
        # Return latest position for each tracked point
        tracks = self.pred_tracks[0, -1].cpu().numpy()  # (N_points, 2)
        tracks_dict = {k: tracks[i][None, :] for i, k in enumerate(self.point_keys)}

        # Add current track to buffer
        if self.pred_tracks_buffer is None:
            self.pred_tracks_buffer = tracks_dict
        else:
            self.pred_tracks_buffer = {k: np.concatenate((self.pred_tracks_buffer[k], tracks_dict[k]), axis=0) for k in self.point_keys}
        return self.pred_tracks_buffer
    
    def step(self, next_image: np.ndarray) -> Optional[Dict[str, np.ndarray]]:

        if self.efficient_mode:
            result = self._step(next_image)
            # Handle warmup
            while result is None:
                result = self._step(next_image)
        else:
            # Pass image model.step times to force predictions at each step
            for i in range(self.model.step):
                result = self._step(next_image)
        return result
    