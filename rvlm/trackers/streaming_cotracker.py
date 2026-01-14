import torch
import numpy as np
from typing import Optional, List, Dict


class StreamingCoTracker:
    """
    Streaming point tracker using CoTracker3 online model.
    Follows similar interface to StreamingDTW: reset() to initialize, step() to process frames.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the streaming tracker.
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online"
        ).to(self.device)
        self.cotracker.eval()
        
        self.step_size = self.cotracker.step  # typically 8
        self.window_len = self.step_size * 2   # typically 16
        
        self.reset()
    
    def reset(self, points_dict: Optional[List[Dict]] = None, frame_shape: Optional[tuple] = None):
        """
        Reset the tracker state. Optionally initialize with points to track.
        
        Args:
            points_dict: List of dicts with 'point' key containing [x_norm, y_norm] in 0-1000 range
                         If None, must be provided on first step() call
            frame_shape: (H, W) of frames to be tracked. Required if points_dict is provided.
        """
        self.frame_buffer: List[np.ndarray] = []
        self.frame_count = 0
        self.is_first_step = True
        self.queries: Optional[torch.Tensor] = None
        self.pred_tracks: Optional[torch.Tensor] = None
        self.pred_visibility: Optional[torch.Tensor] = None
        self.points_dict = points_dict
        self.frame_shape = frame_shape
        
        # Pre-compute queries if points provided
        if points_dict is not None and frame_shape is not None:
            self._setup_queries(points_dict, frame_shape)
    
    def _setup_queries(self, points_dict: List[Dict], frame_shape: tuple):
        """Setup query points for tracking."""
        # H, W = frame_shape[:2]
        # y_norm = [W * p["point"][0] / 1000.0 for p in points_dict]
        # x_norm = [H * p["point"][1] / 1000.0 for p in points_dict]
        
        # x_norm = [p["points"][0,0] for p in points_dict]
        # y_norm = [p["points"][0,1] for p in points_dict]

        x_norm = [v[0] for v in points_dict.values()]
        y_norm = [v[1] for v in points_dict.values()]
        
        t0 = np.zeros(len(points_dict), np.float32)
        
        self.queries = torch.tensor(
            np.stack([t0, x_norm, y_norm], axis=1),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        self.points_dict = points_dict
        self.frame_shape = frame_shape
    
    def _to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Convert list of numpy frames to tensor (1, T, 3, H, W)."""
        t = torch.tensor(np.stack(frames), dtype=torch.float32, device=self.device)
        t = t.permute(0, 3, 1, 2).unsqueeze(0)
        return t
    
    def step(
        self, 
        new_img: np.ndarray, 
        points_dict: Optional[List[Dict]] = None
    ) -> Optional[np.ndarray]:
        """
        Process a new frame and return the entire track history.
        
        Args:
            new_img: numpy array (H, W, 3) - new frame to process
            points_dict: Optional points to track (only used on first frame if not set in reset())
            
        Returns:
            tracks: numpy array (T, N_points, 2) - all tracked positions up to current frame,
                    or None if not enough frames accumulated yet for first prediction
        """
        # Setup queries on first frame if not already done
        if self.queries is None:
            if points_dict is None:
                raise ValueError("points_dict must be provided on first step() or in reset()")
            self._setup_queries(points_dict, new_img.shape)
        
        # Add frame to buffer
        self.frame_buffer.append(new_img)
        self.frame_count += 1
        
        # Process on first frame (with padding) or every step_size frames
        should_process = self.is_first_step or (self.frame_count % self.step_size == 0)
        
        if should_process:
            with torch.inference_mode():
                # Pad if needed, otherwise take last window_len frames
                if len(self.frame_buffer) < self.window_len:
                    pad_count = self.window_len - len(self.frame_buffer)
                    frames = [self.frame_buffer[0]] * pad_count + self.frame_buffer
                else:
                    frames = self.frame_buffer[-self.window_len:]
                video_chunk = self._to_tensor(frames)
                
                if self.is_first_step:
                    # First call initializes internal state
                    self.cotracker(video_chunk, is_first_step=True, queries=self.queries)
                    # Second call gets actual predictions
                    self.pred_tracks, self.pred_visibility = self.cotracker(
                        video_chunk, is_first_step=False, queries=None
                    )
                    self.is_first_step = False
                else:
                    self.pred_tracks, self.pred_visibility = self.cotracker(
                        video_chunk, is_first_step=False, queries=None
                    )
        
        # Return tracks with correct indexing
        if self.pred_tracks is not None:
            tracks = self.pred_tracks[0].detach().cpu().numpy()
            
            if self.frame_count <= self.window_len:
                # Early frames: real frames are at the END due to padding
                tracks = tracks[-self.frame_count:]
            else:
                # After warmup: skip padding period, take real frames
                tracks = tracks[self.window_len:][:self.frame_count]
            
            return {
                k: tracks[:, i] 
                for i, k in enumerate(self.points_dict.keys())
            }
        
        return None

    def finalize(self) -> Optional[np.ndarray]:
        """
        Process any remaining frames that haven't been processed yet.
        Call this after the last step() to ensure all frames are tracked.
        
        Returns:
            tracks: numpy array (T, N_points, 2) - final tracks for all frames
        """
        if self.queries is None or len(self.frame_buffer) == 0:
            return None
        
        remaining = self.frame_count % self.step_size
        
        # Only process if there are unprocessed frames or if we never processed
        if remaining != 0 or self.is_first_step:
            with torch.inference_mode():
                # Get frames for final chunk
                if len(self.frame_buffer) < self.window_len:
                    # Pad if needed
                    pad_count = self.window_len - len(self.frame_buffer)
                    final_frames = [self.frame_buffer[0]] * pad_count + self.frame_buffer
                else:
                    final_frames = self.frame_buffer[-self.window_len:]
                
                video_chunk = self._to_tensor(final_frames)
                
                self.pred_tracks, self.pred_visibility = self.cotracker(
                    video_chunk,
                    is_first_step=self.is_first_step,
                    queries=self.queries if self.is_first_step else None,
                )
                self.is_first_step = False
        
        if self.pred_tracks is not None:
            tracks = self.pred_tracks[0].detach().cpu().numpy()
            return tracks[:self.frame_count]
        
        return None
    
    def get_tracks(self) -> Optional[np.ndarray]:
        """
        Get current tracks without processing new frames.
        
        Returns:
            tracks: numpy array (T, N_points, 2) or None if no predictions yet
        """
        if self.pred_tracks is not None:
            tracks = self.pred_tracks[0].detach().cpu().numpy()
            return tracks[:self.frame_count]
        return None
    
    def get_visibility(self) -> Optional[np.ndarray]:
        """
        Get current visibility scores.
        
        Returns:
            visibility: numpy array (T, N_points) or None if no predictions yet
        """
        if self.pred_visibility is not None:
            vis = self.pred_visibility[0].detach().cpu().numpy()
            return vis[:self.frame_count]
        return None