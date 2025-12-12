import h5py
import numpy as np
from typing import List, Dict, Iterator


class KinBuffer:
    """Buffer class for loading kinematic solutions from HDF5 files."""
    
    def __init__(self, dataset_path: str, keys: List[str]):
        """
        Args:
            dataset_path: Path to the HDF5 dataset file
            keys: List of observation keys to load (e.g., ["kin_K1", "kin_K2"])
        """
        self.dataset_path = dataset_path
        self.keys = keys
        self._load()
    
    def _load(self) -> List[Dict[str, np.ndarray]]:
        """
        Load kinematic solutions for each demo.
        
        Returns:
            List of dictionaries, where each dict represents a demo and contains
            the specified keys with numpy arrays as values.
        """
        self.data = []
        
        with h5py.File(self.dataset_path, "r", swmr=True) as f:
            for dk in f["data"].keys():
                demo = {}
                for key in self.keys:
                    try:
                        demo[key] = f["data"][dk]["obs"][key][:]
                    except KeyError:
                        raise KeyError(f"Key {key} not found. Available keys: {f['data'][dk]['obs'].keys()}")
                self.data.append(demo)
        
    def get_keys(self) -> List[str]:
        return self.keys

    def get_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        data_dict = {}
        for k in self.keys:
            data_dict[k] = [demo[k] for demo in self.data]
        return data_dict

    def __len__(self) -> int:
        return len(self.demos)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return self.demos[index]

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        return iter(self.demos)

    def __next__(self) -> Dict[str, np.ndarray]:
        return next(self.demos)