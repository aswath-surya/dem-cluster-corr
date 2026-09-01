"""Vectorized parallel-schedule sum-product BP (fast front-end).

Same algorithm and conventions as loop_bp.BeliefPropagator (LLR cavity
messages, edge order = np.where(H) row-major), but the per-iteration work is
padded-array numpy instead of Python loops.  Index structure is built once
per H; each run() is a fresh syndrome.

The result object duck-types the BeliefPropagator attributes that
cluster_bp.ClusterBP.load_messages consumes (H, syndrome, p_prior,
llr_b2c_arr, llr_c2b_arr, llr_bit, converged).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

_LLR_CLIP = 20.0


class FastBPResult:
    __slots__ = ("H", "syndrome", "p_prior", "llr_b2c_arr", "llr_c2b_arr",
                 "llr_bit", "converged", "iterations")


class FastBP:
    def __init__(self, H):
        if issparse(H):
            H = H.toarray()
        self.H = np.asarray(H, dtype=np.uint8)
        M, N = self.H.shape
        self.M, self.N = M, N
        rows, cols = np.where(self.H != 0)
        self.n_edges = len(rows)
        self.edge_check, self.edge_bit = rows, cols

        deg_c = np.bincount(rows, minlength=M)
        self.dc_max = int(deg_c.max())
        self.cslots = np.full((M, self.dc_max), -1, dtype=np.int64)
        fill = np.zeros(M, dtype=np.int64)
        for k in range(self.n_edges):
            i = rows[k]
            self.cslots[i, fill[i]] = k
            fill[i] += 1
        self.cmask = self.cslots >= 0
        self.cslots_safe = np.where(self.cmask, self.cslots, 0)

    def run(self, syndrome, p_prior, max_iter=50, tol=1e-6, init_b2c=None):
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        p_prior = np.clip(np.asarray(p_prior, dtype=float), 1e-10, 1 - 1e-10)
        lam = np.log((1.0 - p_prior) / p_prior)
        sgn_s = (1.0 - 2.0 * syndrome.astype(np.float64))     # (M,)

        eb = self.edge_bit
        b2c = lam[eb].copy() if init_b2c is None else np.asarray(init_b2c, dtype=float).copy()
        c2b = np.zeros(self.n_edges)
        llr_bit = lam.copy()
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            old = llr_bit

            t = np.tanh(np.clip(b2c, -_LLR_CLIP, _LLR_CLIP) / 2.0)
            t = np.where(np.abs(t) < 1e-12, np.where(t >= 0, 1e-12, -1e-12), t)
            t_pad = np.where(self.cmask, t[self.cslots_safe], 1.0)
            prod_all = np.prod(t_pad, axis=1) * sgn_s                 # (M,)
            pe = prod_all[:, None] / t_pad
            pe = np.clip(pe, -1 + 1e-12, 1 - 1e-12)
            c2b_pad = 2.0 * np.arctanh(pe)
            c2b[self.cslots_safe[self.cmask]] = c2b_pad[self.cmask]

            S = np.bincount(eb, weights=c2b, minlength=self.N)
            llr_bit = lam + S
            b2c = llr_bit[eb] - c2b

            if np.max(np.abs(llr_bit - old)) < tol:
                converged = True
                break

        r = FastBPResult()
        r.H, r.syndrome, r.p_prior = self.H, syndrome, p_prior
        r.llr_b2c_arr, r.llr_c2b_arr = b2c, c2b
        r.llr_bit, r.converged, r.iterations = llr_bit, converged, it
        return r


__all__ = ["FastBP", "FastBPResult"]
