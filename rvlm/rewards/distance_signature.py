import numpy as np
from .distance_base import StreamingDistanceBase

try:
    import iisignature
except:
    print("WARNING: iisignature not available!")

class StreamingSignature(StreamingDistanceBase):
    """
    Signature Transforms for streaming data. Use to compute signature transforms between multiple series and a (streaming) query.
    """

    def __init__(self, series, normalize_cost: bool = False, m=3):
        """
        series: list of numpy arrays (M, D)
        normalize_cost: whether to normalize the cost by the number of timesteps
        """
        super().__init__(series, normalize_cost)
        
        self.m = m
        self.d = series[0].shape[-1]
        self.sigs_series = [iisignature.sig(s, self.m) for s in series]
        
        self.reset()

    def reset(self):
        """
        Resets the query and cached signature transform.
        """
        self.query = None
        self.sigs_query = None

    def step(self, next_query: np.ndarray):
        """
        Streaming mode (calling step() multiple times): appends the next query to the cached query and computes the costs;
        next_query: numpy array (N, D)
        returns: list of costs (M,): cost for each series
        """
        assert len(next_query.shape) == 2


        if self.query is None:
            self.sigs_query = iisignature.sig(next_query, self.m)
            self.query = next_query
        else:

            disp = (next_query - self.query[-1]).ravel()
            self.sigs_query = iisignature.sigjoin(self.sigs_query, disp, self.m)
            
            self.query = np.concatenate((self.query, next_query), axis=0)

        costs = [np.linalg.norm(s - self.sigs_query) for s in self.sigs_series]
        if self.normalize_cost:
            costs = [float(c) / len(self.query) for c in costs]
    
        return np.array(costs)
