"""Correlated-noise model validation from the command line: inject pair
events into a surface DEM, build the marginal-matched independent twin, and
compare exact / BP / BP+cluster estimates of the per-shot likelihood ratio.
Same code path as the notebook (experiments.py).

    python run_correlated_validation.py                     # phenomenological
    python run_correlated_validation.py --circuit-level     # circuit level
    python run_correlated_validation.py --shots 500 --fig out.png
"""

import argparse

import numpy as np

from experiments import (surface_dem, inject_correlated_pairs,
                         correlated_llr, loop_spectra)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-d", "--distance", type=int, default=3)
    ap.add_argument("-p", type=float, default=None,
                    help="base error rate (default 0.01 phenom, 0.003 circuit)")
    ap.add_argument("--pc", type=float, default=0.005,
                    help="injected correlated-event probability")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--circuit-level", action="store_true")
    ap.add_argument("--no-exact", action="store_true",
                    help="skip the exact reference (needed beyond d=3)")
    ap.add_argument("--fig", default=None,
                    help="save the per-shot LLR figure here")
    args = ap.parse_args()
    p = args.p or (0.003 if args.circuit_level else 0.01)

    _, dem, dt = surface_dem(args.distance, None, p, args.circuit_level)
    twins = inject_correlated_pairs(dem, dt, args.pairs, args.pc)
    noise = "circuit-level" if args.circuit_level else "phenomenological"
    print(f"surface d={args.distance}, {noise}, p={p}: injected "
          f"{len(twins['pairs'])} pair events at pc={args.pc}: "
          f"{twins['pairs']}")

    rows = correlated_llr(twins, n_shots=args.shots,
                          circuit_level=args.circuit_level,
                          exact=not args.no_exact)
    ex, bp, ce = rows[:, 0], rows[:, 1], rows[:, 2]
    print(f"\nmean LLR: exact {ex.mean():+.4f}  BP {bp.mean():+.4f}  "
          f"CE {ce.mean():+.4f}")
    if not args.no_exact:
        print(f"per-shot |err| vs exact: BP {np.abs(bp - ex).mean():.4f}, "
              f"CE {np.abs(ce - ex).mean():.4f}")
    mc, md, is_new = loop_spectra(twins, n_shots=10,
                                  circuit_level=args.circuit_level)
    n_new = sum(1 for k in mc if is_new(k))
    print(f"loops: {len(mc)} in correlated model, {len(md)} in independent "
          f"twin, {n_new} created by the injected events")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from experiments import BLUE, ORANGE, GRAY, INK
        o = np.argsort(ex)
        fig, ax = plt.subplots(figsize=(9, 4.4))
        xs = np.arange(len(ex))
        ax.plot(xs, ex[o], "-", color=INK, lw=1.8, label="exact", zorder=3)
        ax.plot(xs, bp[o], ".", color=GRAY, ms=6, label="BP only")
        ax.plot(xs, ce[o], ".", color=ORANGE, ms=6, label="BP + clusters")
        ax.axhline(0, color=GRAY, lw=0.7)
        ax.set_xlabel("shot (sorted by exact LLR)")
        ax.set_ylabel("per-shot LLR, correlated vs independent")
        ax.legend(frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.fig, dpi=150, bbox_inches="tight")
        print(f"figure saved to {args.fig}")


if __name__ == "__main__":
    main()
