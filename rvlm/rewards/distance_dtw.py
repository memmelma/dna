import numpy as np
from numba import jit
from .distance_base import StreamingDistanceBase

# @jit(nopython=True)
# def compute_acc_cost_matrix(C: np.ndarray, R_cache: np.ndarray = None) -> np.ndarray:
#     """
#     Accumulates cost matrix; supports incremental caching
#     C: cost matrix(N, M)
#     R_cache: cached accumulated cost matrix (N, M) (optional)
#     returns: accumulated cost matrix R: (N+1, M+1) with +inf borders and R[0,0]=0
#     """
#     N, M = C.shape
#     # R has one extra row/col for simpler boundaries; R[0, *] and R[*, 0] start at +inf except R[0,0]=0
#     R = np.full((N + 1, M + 1), np.inf, dtype=C.dtype)
#     R[0, 0] = 0.0

#     N_start, M_start = 1, 1
#     if R_cache is not None:
#         n_cached, m_cached = R_cache.shape
#         R[:n_cached, :m_cached] = R_cache
#         # next new column to fill
#         M_start = m_cached - 1

#     for j in range(M_start, M + 1):
#         # fill column j for all rows 1..N
#         for i in range(N_start, N + 1):
            
#             # # original: minimum of prev acc cost (down, left, diagonal)
#             # m = min(R[i - 1, j], R[i, j - 1], R[i - 1, j - 1])
            
#             # modified: forces query to progress one step
#             m = min(R[i - 1, j], R[i - 1, j - 1])
            
#             # accumulated cost is prev acc cost + curr cost
#             R[i, j] = C[i - 1, j - 1] + m
#     return R


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
def compute_acc_cost_matrix(C, R_cache=None, alpha=0.5, beta=2.0):
    """
    Accumulates cost matrix; supports incremental caching
    
    Itakura parallelogram constraint: only compute if alpha*i <= j <= beta*i
    alpha: minimum slope, beta: maximum slope
    e.g., to allow for up to 0.5 slower and 2x faster than the reference trajectory, set alpha=0.5 and beta=2.0

    C: cost matrix(N, M)
    R_cache: cached accumulated cost matrix (N, M) (optional)
    returns: accumulated cost matrix R: (N+1, M+1) with +inf borders and R[0,0]=0
    """

    N, M = C.shape
    R = np.full((N+1, M+1), np.inf, dtype=C.dtype)
    R[0,0] = 0.0

    # caching logic as you have it...
    N_start, M_start = 1, 1
    if R_cache is not None:
        n_cached, m_cached = R_cache.shape
        R[:n_cached, :m_cached] = R_cache
        M_start = m_cached - 1

    for j in range(M_start, M + 1):
        for i in range(1, N + 1):
            # slope constraint: only compute if alpha*i <= j <= beta*i
            if j < alpha * i or j > beta * i:
                continue

            prev = min(R[i-1, j], R[i, j-1], R[i-1, j-1])
            R[i, j] = C[i-1, j-1] + prev

    return R

class StreamingDTW(StreamingDistanceBase):
    """
    Efficient DTW for streaming data w/ caching support. Use to compute DTW distances between multiple series and a (streaming) query. Supports flexible end (match last timestep to any timestep instead of only the last timestep).
    No implementation of DTW backtracking to get best match, just returns costs and final indices!
    """

    def __init__(self, series, flexible_end: bool = True, normalize_cost: bool = True, alpha: float = 0.0, beta: float = 100.0):
        """
        series: list of numpy arrays (M, D)
        flexible_end: whether to allow the last timestep to be matched to any timestep instead of only the last timestep
        normalize_cost: whether to normalize the cost by the number of timesteps
        """
        super().__init__(series, normalize_cost)
        self.flexible_end = flexible_end
        self.alpha = alpha
        self.beta = beta
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
            self.Rs_new = [compute_acc_cost_matrix(c, alpha=self.alpha, beta=self.beta) for c in Cs_init]

        else:

            self.query = np.concatenate((self.query, next_query), axis=0)

            # compute cost for the new (last) query row
            Cs_partial = [compute_cost_matrix(s.astype(np.float32), self.query[-1:].astype(np.float32)) for s in self.series]
            # append as a new column
            self.Cs_new = [np.concatenate((cn, cp), axis=1) for cn, cp in zip(self.Cs_new, Cs_partial)]
            # accumulated cost for the new cost matrix
            self.Rs_new = [compute_acc_cost_matrix(cn, R_cache=rn, alpha=self.alpha, beta=self.beta) for cn, rn in zip(self.Cs_new, self.Rs_new)]

        if self.flexible_end:
            # open-end DTW / prefix-DTW
            costs = [np.min(rn[1:, -1]) for rn in self.Rs_new]
            idcs = [np.argmin(rn[1:, -1]) for rn in self.Rs_new]
        else:
            costs = [rn[-1, -1] for rn in self.Rs_new]
            idcs = [-1] * len(costs)

        if self.normalize_cost:
            costs = [float(c) / len(self.query) for c in costs]

        return np.array(costs)

    def step_package(self, next_query: np.ndarray):
        """
        Streaming mode using dtw package instead of custom cost matrix computation.
        Uses the same parameters as the notebook: asymmetric step pattern, open begin/end.
        
        Args:
            next_query: numpy array (N, D) - next frame(s) to append to query
        
        Returns:
            numpy array (M,) - normalized DTW distances for each series
        """
        from dtw import dtw
        
        assert len(next_query.shape) == 2, "next_query must be 2D array (N, D)"
        
        # Update query: append new frames to cached query
        if self.query is None:
            self.query = next_query
        else:
            self.query = np.concatenate((self.query, next_query), axis=0)
        
        # Compute DTW for each series using the dtw package
        costs = []
        for series in self.series:
            alignment = dtw(
                x=series,              # reference trajectory
                y=self.query,          # query trajectory (growing)
                step_pattern='asymmetric',
                open_begin=True,
                open_end=True,
                keep_internals=True
            )
            
            # Use normalized distance (equivalent to notebook usage)
            distance = alignment.normalizedDistance
            costs.append(distance)
        
        return np.array(costs)

    def forward(self, query: np.ndarray):
        """
        Non-streaming version: computes DTW costs for entire trajectory at once.
        Returns costs for each timestep of the query trajectory.
        
        query: numpy array (T, D) - entire trajectory
        returns: numpy array (M, T) - costs for each series at each timestep
        """
        assert len(query.shape) == 2, "Query must be 2D array (T, D)"
        
        T = len(query)
        M = len(self.series)
        all_costs = np.zeros((M, T), dtype=np.float32)
        
        for t in range(T):
            # Get query up to current timestep
            query_t = query[:t+1]
            
            # Compute cost matrix for each series
            Cs = [compute_cost_matrix(s.astype(np.float32), query_t.astype(np.float32)) for s in self.series]
            
            # Compute accumulated cost matrix
            Rs = [compute_acc_cost_matrix(c, alpha=self.alpha, beta=self.beta) for c in Cs]
            
            # Extract costs based on flexible_end setting
            if self.flexible_end:
                # Open-end DTW: minimum cost across all series endpoints
                costs_t = [np.min(r[1:, -1]) for r in Rs]
            else:
                # Fixed-end DTW: cost at final alignment
                costs_t = [r[-1, -1] for r in Rs]
            
            # Normalize if requested
            if self.normalize_cost:
                costs_t = [float(c) / (t + 1) for c in costs_t]
            
            all_costs[:, t] = costs_t
        
        return all_costs