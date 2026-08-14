# Detector error models as tensor networks

Evaluates syndrome-history probabilities of stim detector error models (DEMs) by
tensor network contraction, exactly where affordable and by belief propagation
with a connected cluster expansion beyond that. Built on quimb, cotengra and
stim. The scientific throughline: the contraction is the forward map of the
Blume-Kohout and Young DEM-estimation framework (arXiv:2504.14643), which makes
it an instrument for maximum-likelihood decoding, noise-model validation, and
locating correlated noise.

## Contents

| file | what it is |
|---|---|
| `tensor_probability_estimation.ipynb` | the main narrative: construction, validation, cluster expansion, correlated-noise experiments, surface codes |
| `dem_tn.py` | DEM to tensor network builder: COPY/XOR construction, observable legs, GF(2) admissibility guard, parity-chain decomposition for high-degree detectors, correlated-event injection with marginal-matched twins |
| `cluster_expansion.py` | hand implementation of the connected cluster expansion of Midha and Zhang (arXiv:2510.02290): edge-subset loop enumeration, excitation projectors, Ursell weights, on top of quimb's D1BP fixed point |
| `dem_viz.py` | figure library (convergence, loop decay, correlation spectra, network drawings) |
| `experiments.py` | experiment drivers and figure functions behind the slide deck; the notebook's reproduction section and both scripts call these |
| `run_surface_ce.py` | command line: exact vs BP vs cluster expansion on a surface-code DEM (`python run_surface_ce.py -d 5 --circuit-level ...`) |
| `run_correlated_validation.py` | command line: correlated-noise twin test with per-shot likelihood ratios (`python run_correlated_validation.py --circuit-level`) |
| `figures/` | all generated figures |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name dem-tn
# open tensor_probability_estimation.ipynb and run top to bottom,
# or reproduce the headline figures without the notebook:
python run_surface_ce.py
python run_correlated_validation.py --fig figures/llr.png
```

## Headline results

- Exact contraction validated against brute force to 1e-15, including logical
  observable legs (maximum-likelihood decoding as a byproduct).
- The connected cluster expansion converges exponentially in loop weight on
  DEM networks; quimb's built-in loop routines do not compute this object.
- Injected correlated noise appears as specific new loops, and the loop
  corrections carry the correlation information a BP-only likelihood misses
  (14 percent of the per-shot log-likelihood ratio on average).
- On 3D phenomenological surface-code networks the expansion is the cheapest
  reliable evaluator: at d=5 it matches compressed contraction to 1e-4 nats at
  30-200x less cost per shot, with both method families certifying each other.
- At circuit level the global expansion is outside its convergence regime
  (spacetime hub detectors create a dense gas of overlapping short loops) and
  plateaus at 1e-2 nats; compressed contraction there can false-plateau below
  the exact product lower bound and needs independent validation. For
  difference objects (model-validation LLRs, observable ratios) a localized
  expansion enumerating only loops through the modified region remains sound.

Key references: Blume-Kohout and Young arXiv:2504.14643; Derks, Townsend-Teague,
Burchards, Eisert arXiv:2407.13826; Midha and Zhang arXiv:2510.02290; Midha,
Sommers, Tindall, Abanin arXiv:2604.03228.
