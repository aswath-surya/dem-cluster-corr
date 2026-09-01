r"""Relay-BP (Müller et al., arXiv:2506.01779) — vectorized implementation.

DMem-BP: min-sum BP with per-error-node memory strengths γ_j applied to the
bias term,
    Λ_j(t) = (1-γ_j) Λ_j(0) + γ_j M_j(t-1),
messages (Eqs. 1-2 of the paper):
    μ_{i→j}(t) = κ_{i,j}(t) (-1)^{σ_i} min_{j'∈N(i)\j} |ν_{j'→i}(t-1)|
    ν_{j→i}(t) = Λ_j(t) + Σ_{i'∈N(j)\i} μ_{i'→j}(t)
    M_j(t)     = Λ_j(t) + Σ_{i∈N(j)} μ_{i→j}(t)

Relay-BP-S (Algorithm 1): chain DMem-BP legs; leg r restarts messages and
bias from the prior but initializes marginals M_j(0) from the previous leg's
final marginals; each leg draws fresh disorder γ_j(r).  Collect up to S
syndrome-consistent solutions across ≤ R legs and return the minimum-weight
one, w(ê) = Σ_j ê_j log((1-p_j)/p_j).

The min-sum check update is vectorized over all checks via padded
(n_checks, max_degree) arrays; a decode costs O(iterations · edges).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import issparse


class RelayBP:
    def __init__(self, H, p_prior, S: int = 1, R: int = 60,
                 T_first: int = 80, T_leg: int = 60,
                 gamma_first: float = 0.125,
                 gamma_range: Tuple[float, float] = (-0.24, 0.66),
                 seed: int = 0):
        if issparse(H):
            H = H.toarray()
        self.H = np.asarray(H, dtype=np.uint8)
        self.M_checks, self.N_bits = self.H.shape
        self.p_prior = np.clip(np.asarray(p_prior, dtype=float), 1e-12, 0.5 - 1e-12)
        self.lam = np.log((1.0 - self.p_prior) / self.p_prior)   # λ_j
        self.S, self.R = S, R
        self.T_first, self.T_leg = T_first, T_leg
        self.gamma_first = gamma_first
        self.gamma_range = gamma_range
        self.rng = np.random.default_rng(seed)

        rows, cols = np.where(self.H != 0)
        self.n_edges = len(rows)
        self.edge_check = rows          # i of edge k
        self.edge_bit = cols            # j of edge k

        # padded per-check edge table for the min-sum update
        deg_c = np.bincount(rows, minlength=self.M_checks)
        self.dc_max = int(deg_c.max())
        self.check_slots = np.full((self.M_checks, self.dc_max), -1, dtype=np.int64)
        fill = np.zeros(self.M_checks, dtype=np.int64)
        for k in range(self.n_edges):
            i = rows[k]
            self.check_slots[i, fill[i]] = k
            fill[i] += 1
        self.check_mask = self.check_slots >= 0
        self.check_slots_safe = np.where(self.check_mask, self.check_slots, 0)

    # ------------------------------------------------------------------
    def _run_leg(self, syndrome_sign, gamma, T, M_init):
        """One DMem-BP leg.  Returns (converged, e_hat, M_final, iters)."""
        lam = self.lam
        nu = lam[self.edge_bit].copy()          # ν_{j→i}(0) = λ_j
        Lam0 = lam                              # Λ_j(0) = λ_j
        M_prev = M_init
        eb = self.edge_bit
        Hs = self.H.astype(np.int64)

        for t in range(1, T + 1):
            Lam = (1.0 - gamma) * Lam0 + gamma * M_prev

            # --- check update (min-sum, product-except-one) ---
            nu_pad = nu[self.check_slots_safe]                    # (M, dc_max)
            absn = np.where(self.check_mask, np.abs(nu_pad), np.inf)
            sgn = np.where(self.check_mask, np.where(nu_pad < 0, -1.0, 1.0), 1.0)
            # two smallest magnitudes per check
            part = np.partition(absn, 1, axis=1)
            min1 = part[:, 0]
            min2 = part[:, 1]
            argmin = np.argmin(absn, axis=1)
            total_sign = np.prod(sgn, axis=1) * syndrome_sign      # (M,)
            is_min = np.arange(self.dc_max)[None, :] == argmin[:, None]
            mags = np.where(is_min, min2[:, None], min1[:, None])
            mu_pad = (total_sign[:, None] * sgn) * mags            # ÷sgn == ×sgn
            mu = np.empty_like(nu)
            mu[self.check_slots_safe[self.check_mask]] = mu_pad[self.check_mask]

            # --- bit update + marginals ---
            S_bit = np.bincount(eb, weights=mu, minlength=self.N_bits)
            M = Lam + S_bit
            nu = M[eb] - mu

            e_hat = (M < 0).astype(np.uint8)
            if np.array_equal((Hs @ e_hat) % 2, self._syn):
                return True, e_hat, M, t, nu
            M_prev = M
        return False, e_hat, M, T, nu

    # ------------------------------------------------------------------
    def decode(self, syndrome: np.ndarray,
               return_info: bool = False):
        """Relay-BP-S decode.  Returns ê (best solution found, else the last
        hard decision), optionally with (found, n_legs, total_iters)."""
        self._syn = np.asarray(syndrome, dtype=np.int64)
        syndrome_sign = 1.0 - 2.0 * self._syn                        # (M,)

        best_e = None
        best_w = np.inf
        best_nu = None
        n_found = 0
        total_iters = 0
        M = self.lam.copy()                     # M_j(0) for the first leg

        for r in range(self.R):
            if r == 0:
                gamma = np.full(self.N_bits, self.gamma_first)
                T = self.T_first
            else:
                lo, hi = self.gamma_range
                gamma = self.rng.uniform(lo, hi, size=self.N_bits)
                T = self.T_leg
            conv, e_hat, M, it, nu = self._run_leg(syndrome_sign, gamma, T, M)
            total_iters += it
            if conv:
                w = float(e_hat @ self.lam)
                n_found += 1
                if w < best_w:
                    best_w, best_e, best_nu = w, e_hat.copy(), nu.copy()
                if n_found >= self.S:
                    break

        out = best_e if best_e is not None else e_hat
        self.last_state = dict(found=n_found > 0, legs=r + 1,
                               iters=total_iters,
                               nu=best_nu if best_nu is not None else nu)
        if return_info:
            return out, (n_found > 0, r + 1, total_iters)
        return out


__all__ = ["RelayBP"]
