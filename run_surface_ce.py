"""Surface-code DEM likelihoods: exact vs BP vs cluster expansion, from the
command line. Same code path as the notebook (experiments.py).

    python run_surface_ce.py                       # d=3 phenomenological
    python run_surface_ce.py -d 5 --shots 3        # d=5 (no exact, width 45)
    python run_surface_ce.py --circuit-level       # circuit-level noise
"""

import argparse

import numpy as np

from dem_tn import contract_exact, make_optimizer
from cluster_expansion import ClusterExpansion
from experiments import surface_dem, build_closed, sample_syndromes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-d", "--distance", type=int, default=3)
    ap.add_argument("-r", "--rounds", type=int, default=None)
    ap.add_argument("-p", type=float, default=0.01)
    ap.add_argument("--circuit-level", action="store_true")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("-M", type=int, nargs="+", default=[4, 6, 8],
                    help="cluster weights to evaluate")
    ap.add_argument("--max-exact-width", type=float, default=30,
                    help="skip exact contraction above this width")
    args = ap.parse_args()

    circuit, dem, dt = surface_dem(args.distance, args.rounds, args.p,
                                   args.circuit_level)
    noise = "circuit-level" if args.circuit_level else "phenomenological"
    print(f"surface d={args.distance}, {noise}, p={args.p}: "
          f"{dt.num_events} mechanisms, {dem.num_detectors} detectors")

    tn0 = build_closed(dt, np.zeros(dem.num_detectors, dtype=np.uint8))
    width = tn0.contraction_tree(optimize="greedy").contraction_width()
    do_exact = width <= args.max_exact_width
    print(f"greedy contraction width {width:.0f} "
          f"({'exact available' if do_exact else 'exact skipped'})")

    opt = make_optimizer(max_repeats=16)
    loops = None
    for si, syn in enumerate(sample_syndromes(circuit, args.shots)):
        tn = build_closed(dt, syn)
        line = f"shot {si} (wt {int(syn.sum())}):"
        lex = None
        if do_exact:
            lex = np.log(contract_exact(tn, optimize=opt))
            line += f" exact {lex:.4f} |"
        ce = ClusterExpansion(tn)
        lbp = np.log(abs(ce.contract_bp()))
        line += (f" BP err {abs(lbp - lex):.2e}" if lex is not None
                 else f" BP {lbp:.4f}")
        if loops is None:
            loops = ce.gen_loops(max(args.M))
            print(f"[{len(loops)} loops up to weight {max(args.M)}, "
                  "shared across shots]")
        ce.gen_loops = lambda mw: [F for F in loops if len(F) <= mw]
        for M in args.M:
            v = np.log(abs(ce.contract(M)))
            line += (f" | CE M={M} err {abs(v - lex):.2e}" if lex is not None
                     else f" | CE M={M} {v:.4f}")
        print(line)


if __name__ == "__main__":
    main()
