"""quimb tensor-network evaluator for stim DetectorErrorModels (detector picture).

Bernoulli-per-event x COPY-broadcast x parity(XOR) construction, following
Piveteau-Chubb-Renes / Blume-Kohout & Young. Adds: proper qtn.COPY_tensor
event nodes, free observable legs, a GF(2) inadmissible-syndrome guard,
cached cotengra exact contraction, and a D1BP + generalized-loop (gloop)
cluster-expansion corrector localized around the observable.
"""
from itertools import product

import numpy as np
import stim
import quimb.tensor as qtn
import cotengra as ctg
from quimb.tensor.belief_propagation import D1BP


def with_correlated_events(dem: stim.DetectorErrorModel, events):
    """Append correlated multi-detector error events to a DEM, and build the
    marginal-matched decorrelated counterpart.

    `events` is a list of `(p, [d1, d2, ...])`: a single error mechanism of
    probability `p` flipping all listed detectors together. Returns
    `(dem_corr, dem_decorr)` where

    - `dem_corr`   has `error(p) D_{d1} D_{d2} ...` appended per entry;
    - `dem_decorr` instead appends the independent singles
      `error(p) D_{d1}`, `error(p) D_{d2}`, ... .

    Because attenuations add under aggregation (BK&Y Eq. 22), every
    single-detector polarization <z_i> is *identical* between the two DEMs:
    each listed detector's decay factor gains the same (1-2p) either way.
    Only multi-detector parities differ -- e.g. <z_i z_j> is untouched by the
    correlated event (its two flips cancel in the parity) but attenuated by
    (1-2p)^2 in the decorrelated model. The pair is exactly Blume-Kohout &
    Young's coincident-vs-correlated degeneracy example, made concrete.
    """
    corr_lines = []
    decorr_lines = []
    for p, dets in events:
        targets = " ".join(f"D{d}" for d in dets)
        corr_lines.append(f"error({p}) {targets}")
        for d in dets:
            decorr_lines.append(f"error({p}) D{d}")
    base = str(dem.flattened())
    dem_corr = stim.DetectorErrorModel(base + "\n" + "\n".join(corr_lines))
    dem_decorr = stim.DetectorErrorModel(base + "\n" + "\n".join(decorr_lines))
    return dem_corr, dem_decorr


def xor_tensor(num_vars, target_bit):
    shape = (2,) * num_vars
    data = np.zeros(shape)
    for bits in product((0, 1), repeat=num_vars):
        if sum(bits) % 2 == target_bit:
            data[bits] = 1.0
    return data


def add_parity(tn, inds, tag, target_bit=None, open_ind=None, chain_cutoff=8):
    """Add a parity constraint over `inds` to `tn`, either projected onto
    `target_bit` or with a free output leg `open_ind`.

    For len(inds) <= chain_cutoff a single dense XOR tensor is used (as
    before). Above that -- e.g. surface-code detectors, whose degree reaches
    48+ with decompose_errors=False, where a dense tensor would need 2^48
    entries -- the parity is decomposed exactly into a chain of degree-3 XOR
    tensors carrying a running-parity accumulator (memory O(k) instead of
    O(2^k), and the TN max degree drops to 3). All chain tensors carry `tag`.
    """
    k = len(inds)
    assert k >= 1
    if open_ind is None:
        if k + 0 <= chain_cutoff:
            tn |= qtn.Tensor(xor_tensor(k, int(target_bit)), inds=tuple(inds),
                             tags={tag})
            return
    else:
        if k + 1 <= chain_cutoff:
            tn |= qtn.Tensor(xor_tensor(k + 1, 0),
                             inds=tuple(inds) + (open_ind,), tags={tag})
            return
    # chain: a_1 = i_1 xor i_2; a_j = a_{j-1} xor i_{j+1}; close on the last
    aux = [f"{tag}_chain{j}" for j in range(k - 2)]
    tn |= qtn.Tensor(xor_tensor(3, 0), inds=(inds[0], inds[1], aux[0]),
                     tags={tag})
    for j in range(1, k - 2):
        tn |= qtn.Tensor(xor_tensor(3, 0), inds=(aux[j - 1], inds[j + 1], aux[j]),
                         tags={tag})
    if open_ind is None:
        tn |= qtn.Tensor(xor_tensor(2, int(target_bit)),
                         inds=(aux[-1], inds[-1]), tags={tag})
    else:
        tn |= qtn.Tensor(xor_tensor(3, 0), inds=(aux[-1], inds[-1], open_ind),
                         tags={tag})


def _gf2_rank(mat):
    m = mat.copy().astype(np.uint8) % 2
    rows, cols = m.shape
    rank = 0
    for col in range(cols):
        piv = next((r for r in range(rank, rows) if m[r, col]), None)
        if piv is None:
            continue
        m[[rank, piv]] = m[[piv, rank]]
        for r in range(rows):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]
        rank += 1
        if rank == rows:
            break
    return rank


class DemTN:
    """Detector-picture TN builder for a stim DEM, with observable legs and
    a GF(2) admissibility guard."""

    def __init__(self, dem: stim.DetectorErrorModel):
        dem = dem.flattened()
        self.dem = dem
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

        self.probs = []
        self.event_dets = []
        self.event_obs = []
        for instr in dem:
            if instr.type != "error":
                continue
            p = instr.args_copy()[0]
            targets = instr.targets_copy()
            self.probs.append(p)
            self.event_dets.append([t.val for t in targets if t.is_relative_detector_id()])
            self.event_obs.append([t.val for t in targets if t.is_logical_observable_id()])

        self.num_events = len(self.probs)

        self.E = np.zeros((self.num_detectors, self.num_events), dtype=np.uint8)
        for i, dets in enumerate(self.event_dets):
            for d in dets:
                self.E[d, i] ^= 1

    def is_admissible(self, syndrome):
        x = np.asarray(syndrome, dtype=np.uint8)
        aug = np.concatenate([self.E, x[:, None]], axis=1)
        return _gf2_rank(self.E) == _gf2_rank(aug)

    def build(self, syndrome, observables=None):
        """Build the TN for a fixed detector syndrome. `observables` maps
        observable index -> 0/1 to project onto; any observable index not
        given as a key is left open as a free index 'obs{k}'.
        Returns None if the syndrome is inadmissible (Pr == 0 exactly)."""
        assert len(syndrome) == self.num_detectors
        if not self.is_admissible(syndrome):
            return None
        observables = observables or {}

        tn = qtn.TensorNetwork([])
        det_incident = {d: [] for d in range(self.num_detectors)}
        obs_incident = {k: [] for k in range(self.num_observables)}

        for i, p in enumerate(self.probs):
            dets = self.event_dets[i]
            obs = self.event_obs[i]
            aux = [f"e{i}_d{d}_{j}" for j, d in enumerate(dets)] + \
                  [f"e{i}_o{k}_{j}" for j, k in enumerate(obs)]
            if not aux:
                continue  # event touches nothing external; marginalizes to 1
            main = f"e{i}_main"
            tn |= qtn.Tensor(np.array([1 - p, p]), inds=(main,), tags={f"event{i}"})
            tn |= qtn.COPY_tensor(2, inds=[main] + aux, tags={f"copy{i}"})
            for j, d in enumerate(dets):
                det_incident[d].append(aux[j])
            for j, k in enumerate(obs):
                obs_incident[k].append(aux[len(dets) + j])

        for d in range(self.num_detectors):
            inds = det_incident[d]
            bit = int(syndrome[d])
            if not inds:
                assert bit == 0  # guaranteed by is_admissible above
                continue
            add_parity(tn, inds, f"det{d}", target_bit=bit)

        self.obs_tags = {}
        for k in range(self.num_observables):
            inds = obs_incident[k]
            if not inds:
                continue
            tag = f"obs{k}"
            if k in observables:
                add_parity(tn, inds, tag, target_bit=int(observables[k]))
            else:
                add_parity(tn, inds, tag, open_ind=tag)
            self.obs_tags[k] = tag

        return tn


def make_optimizer(cache_dir=None, max_repeats=64):
    """A single reusable optimizer object amortizes path search across many
    contractions of TNs sharing the same index structure (e.g. many
    syndromes for the same DEM) -- pass the *same* instance to repeated
    `contract_exact` calls."""
    return ctg.ReusableHyperOptimizer(
        methods=["greedy"],
        max_repeats=max_repeats,
        parallel=False,
        progbar=False,
        minimize="combo",
        directory=cache_dir,
    )


def contract_exact(tn, output_inds=(), optimize=None):
    if optimize is None:
        optimize = make_optimizer()
    return tn.contract(output_inds=output_inds, optimize=optimize)


def contract_bp_gloop_near_observable(tn, obs_tag, max_size=6, bp_opts=None, gloop_opts=None):
    """Approximate contraction of a *closed* TN (syndrome + observable both
    projected to fixed bits) via D1BP, corrected by a truncated loop-series
    expansion (Gray & Kourtis-style / arXiv:2409.03108) restricted to
    generalized loops ('gloops') that pass through the tensor carrying
    `obs_tag` -- i.e. the cluster correction is spent where the decoding
    decision is actually made, rather than uniformly over the whole DEM.

    Note our DEM TN is already a *simple* graph TN, not a hypergraph one:
    every index appears on exactly 2 tensors because each event's broadcast
    is materialized as an explicit `qtn.COPY_tensor` rather than left as a
    native hyperindex. So D1BP (dense, 1-norm, no-hyperindex BP) applies
    directly with no `hyperinds_resolve` step needed, and its
    `contract_loop_series_expansion` accepts an arbitrary, possibly
    incomplete, set of loops as a legitimate (if partial) additive
    correction to the Bethe free energy -- unlike HD1BP's Kikuchi-style
    `contract_gloop_expand`, which requires a *complete* covering set of
    loops for its region-counting algebra to cancel correctly, and silently
    returns garbage if fed a localized subset.
    """
    bp_opts = bp_opts or {}
    gloop_opts = gloop_opts or {}
    bp = D1BP(tn, **bp_opts)
    bp.run()
    bp_val = bp.contract()
    obs_tids = [tid for tid, t in tn.tensor_map.items() if obs_tag in t.tags]
    gloops = tuple(
        tn.gen_gloops(max_size=max_size, tids=obs_tids, grow_from="any", **gloop_opts)
    )
    if not gloops:
        # region too small to contain any loop through the observable at
        # this max_size -- no correction available, report plain BP
        return bp, gloops, bp_val
    try:
        val = bp.contract_loop_series_expansion(gloops=gloops, optimize="greedy")
    except OverflowError:
        # self-consistent multi-excitation iteration diverged -- typically
        # happens deep in the tail (Pr ~ 1e-8 and below) where the starting
        # BP estimate is already off by orders of magnitude; fall back to
        # the single-excitation (non-self-consistent) loop correction
        val = bp.contract_loop_series_expansion(
            gloops=gloops, optimize="greedy", multi_excitation_correct=False
        )
    return bp, gloops, val
