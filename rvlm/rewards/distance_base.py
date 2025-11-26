from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union, Tuple


class StreamingDistanceBase(ABC):
    """
    Abstract base class for streaming distance computation between multiple series and a streaming query.
    
    Subclasses should implement:
    - __init__: Initialize with series and configuration parameters
    - reset: Reset the query and cached data
    - step: Process next query chunk and return costs
    """

    def __init__(self, series: List[np.ndarray], normalize_cost: bool = False):
        """
        Initialize the streaming distance computation.
        
        Args:
            series: List of numpy arrays, each of shape (M, D)
            normalize_cost: Whether to normalize the cost by the number of timesteps
        """
        self.series = series
        self.normalize_cost = normalize_cost
        self.query = None
        self.reset()

    @abstractmethod
    def reset(self):
        """
        Resets the query and cached data structures.
        Should be called before processing a new query sequence.
        """
        pass

    @abstractmethod
    def step(self, next_query: np.ndarray) -> Union[List[float], Tuple[List[float], List[int]]]:
        """
        Process the next query chunk and compute costs.
        
        Can be called once for batch processing or multiple times for streaming.
        
        Args:
            next_query: Numpy array of shape (N, D) representing the next query chunk
            
        Returns:
            Either:
            - List of costs (one per series)
            - Tuple of (costs, indices) where indices indicate best matching positions
        """
        pass

