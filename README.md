# Detector error models as tensor networks

Syndrome-history probabilities for stim detector error models, by tensor-network
contraction: exact where the contraction width allows, belief propagation plus a
connected cluster expansion (Midha & Zhang, arXiv:2510.02290) also present. Relay-BP added as contraction method although notebook doesn't incorporate this yet. The
contraction is the forward map of the DEM-estimation framework of Blume-Kohout &
Young (arXiv:2504.14643), so with this one object we can do maximum-likelihood decoding (MLD),
noise-model validation, and locating correlated noise. MLD, however, requires additional effort (ongoing): specifically, for qLDPC codes, logical classes $\sim 2^{10}$ so identifying Pr $(l | \vec{s})$ quite difficult; worm algorithm?

The code is built on stim, quimb and cotengra.

## Layout

`tensor_probability_estimation.ipynb` is a walkthrough: construction,
validation, the cluster expansion, correlated-noise experiments, surface codes.
The main machinery is in `dem_tn.py` (DEM to network: COPY/XOR construction,
observable legs, GF(2) admissibility/parity evaluation, parity-chain decomposition for
high-degree detectors, correlated-event injection with marginal-matched twins)
and `cluster_expansion.py` (edge-subset loop enumeration, excitation
projectors, Ursell weights, on quimb's D1BP fixed point). `dem_viz.py` draws
the figures, `experiments.py` has drivers behind notebook's
reproduction section & other scripts:

```bash
python run_surface_ce.py -d 5            # exact vs BP vs cluster expansion
python run_correlated_validation.py      # twin-model correlation test
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run the notebook top to bottom, or the two scripts above.

## What came out

Exact contraction agrees with brute force at the 1e-15 level, observable legs included, which makes the evaluator an ML decoder as well (after sorting out the logical class identification problem). The cluster expansion converges exponentially in loop weight on these networks (quimb's built-in loop routines compute a different object and do not). Injected correlated noise shows up as specific new loops, and the loop corrections carry the correlation information a BP-only likelihood misses: O(10) percent of the per-shot log-likelihood ratio on average, more on the shots that matter.

On 3D phenomenological surface-code networks the expansion is the cheapest reliable evaluator: at d=5 it matches compressed contraction to 1e-4 at 30-200x less cost per shot, and the two method (families) certify each other. At circuit level the global expansion leaves its convergence regime (spacetime hub detectors make a dense gas of overlapping short loops) and plateaus at 1e-2 (LLR), while compressed contraction there can false-plateau below an exact lower bound, so neither is trustworthy alone. Note that clearly for reasonably large 3D surface codes, exact contraction quickly becomes infeasible. Difference objects, model-validation LLRs and observable ratios, stay tractable at circuit level through a localized expansion that only enumerates loops through the modified region (consult 2604.03228 treatment of observables/two-point correlators).

## References

Blume-Kohout & Young, arXiv:2504.14643. Derks, Townsend-Teague, Burchards,
Eisert, arXiv:2407.13826. Midha & Zhang, arXiv:2510.02290. Midha, Sommers,
Tindall, Abanin, arXiv:2604.03228.
