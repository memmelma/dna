import numpy as np
from numba import jit
from .distance_base import StreamingDistanceBase

# # @jit(nopython=True)
def compute_cost_matrix(series: np.ndarray, query: np.ndarray, p: int = 2) -> np.ndarray:
    """
    Pairwise distances using numpy broadcasting
    series: (N,D)
    query: (M,D)
    returns: cost matrix C: (N,M)
    """
    s = series.reshape(len(series), -1)     # (N, D)
    q = query.reshape(len(query), -1)       # (M, D)
    diff = s[:, None, :] - q[None, :, :]    # (N, M, D)
    return np.power(np.sum(np.abs(diff) ** p, axis=2), 1.0 / p)
    
@jit(nopython=True)
def compute_acc_cost_matrix(C: np.ndarray, R_cache: np.ndarray = None) -> np.ndarray:
    """
    Accumulates cost matrix; supports incremental caching
    C: cost matrix(N, M)
    R_cache: cached accumulated cost matrix (N, M) (optional)
    returns: accumulated cost matrix R: (N+1, M+1) with +inf borders and R[0,0]=0
    """
    N, M = C.shape
    # R has one extra row/col for simpler boundaries; R[0, *] and R[*, 0] start at +inf except R[0,0]=0
    R = np.full((N + 1, M + 1), np.inf, dtype=C.dtype)
    R[0, 0] = 0.0

    N_start, M_start = 1, 1
    if R_cache is not None:
        n_cached, m_cached = R_cache.shape
        R[:n_cached, :m_cached] = R_cache
        # next new column to fill
        M_start = m_cached - 1

    for j in range(M_start, M + 1):
        # fill column j for all rows 1..N
        for i in range(N_start, N + 1):
            
            # # original: minimum of prev acc cost (down, left, diagonal)
            # m = min(R[i - 1, j], R[i, j - 1], R[i - 1, j - 1])
            
            # modified: forces query to progress one step
            m = min(R[i - 1, j - 1], R[i - 1, j])
            
            # accumulated cost is prev acc cost + curr cost
            R[i, j] = C[i - 1, j - 1] + m
    return R

class StreamingDTW(StreamingDistanceBase):
    """
    Efficient DTW for streaming data w/ caching support. Use to compute DTW distances between multiple series and a (streaming) query. Supports flexible end (match last timestep to any timestep instead of only the last timestep).
    No implementation of DTW backtracking to get best match, just returns costs and final indices!
    """

    def __init__(self, series, flexible_end: bool = True, normalize_cost: bool = True):
        """
        series: list of numpy arrays (M, D)
        flexible_end: whether to allow the last timestep to be matched to any timestep instead of only the last timestep
        normalize_cost: whether to normalize the cost by the number of timesteps
        """
        super().__init__(series, normalize_cost)
        self.flexible_end = flexible_end
        self.reset()

    def reset(self):
        """
        Resets the query and cached cost and accumulated cost matrices.
        """
        self.query = None
        self.Cs_new, self.Rs_new = None, None

    def step(self, next_query: np.ndarray):
        """
        Regular mode (calling step() once): computes the costs and indices for the entire query; call reset() to start a new query
        Streaming mode (calling step() multiple times): appends the next query to the cached query and computes the costs and indices; caches the cost and accumulated cost matrices
        next_query: numpy array (N, D)
        returns: list of costs (M,), list of indices (M,): cost and index for each series
        """
        assert len(next_query.shape) == 2

        if self.query is None:
            self.query = next_query

            # compute cost for first query
            Cs_init = [compute_cost_matrix(s.astype(np.float32), self.query.astype(np.float32)) for s in self.series]
            self.Cs_new = Cs_init
            # accumulated cost for first query
            self.Rs_new = [compute_acc_cost_matrix(c) for c in Cs_init]

        else:

            self.query = np.concatenate((self.query, next_query), axis=0)

            # compute cost for the new (last) query row
            Cs_partial = [compute_cost_matrix(s.astype(np.float32), self.query[-1:].astype(np.float32)) for s in self.series]
            # append as a new column
            self.Cs_new = [np.concatenate((cn, cp), axis=1) for cn, cp in zip(self.Cs_new, Cs_partial)]
            # accumulated cost for the new cost matrix
            self.Rs_new = [compute_acc_cost_matrix(cn, R_cache=rn) for cn, rn in zip(self.Cs_new, self.Rs_new)]

        if self.flexible_end:
            # costs = [np.min(rn[:, -1]) for rn in self.Rs_new]
            # idcs = [np.argmin(rn[:, -1]) for rn in self.Rs_new]
            costs = [np.min(rn[-1:]) for rn in self.Rs_new]
            idcs = [np.argmin(rn[-1,:]) for rn in self.Rs_new]
        else:
            costs = [rn[-1, -1] for rn in self.Rs_new]
            idcs = [-1] * len(costs)

        if self.normalize_cost:
            costs = [float(c) / len(self.query) for c in costs]

        # print(len(self.query))
        # data = {
        #     "query": self.query,
        #     "series": self.series,
        #     "costs": costs,
        #     "Rs_new": self.Rs_new,
        #     "Cs_new": self.Cs_new,
        #     "idcs": idcs,
        # }
        # import pickle
        # with open("dtw_cache_data.pkl", "wb") as f:
        #     pickle.dump(data, f)

        return costs, idcs