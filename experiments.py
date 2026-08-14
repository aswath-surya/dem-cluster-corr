"""Reusable experiment drivers behind the summary figures.

Everything here wraps DemTN and ClusterExpansion without modifying them. The
run_* functions return plain arrays so that the notebook and the command-line
scripts (run_surface_ce.py, run_correlated_validation.py) share one code
path; the fig_* functions turn those arrays into the figures used in
dem_results_slides. Seeds are fixed throughout so every number is
reproducible: the sampler seed 42 draws the shots, seed 7 draws test
syndromes, and all contractions are deterministic.

Conventions repeated from the rest of the package: networks are built closed
(observable legs capped with the all-ones vector, i.e. logical classes
marginalized), phenomenological noise means per-round data depolarization
plus measurement flips, and circuit-level noise adds gate and reset faults.
"""

import io

import numpy as np
import stim
import quimb.tensor as qtn
import matplotlib.pyplot as plt

from dem_tn import DemTN, contract_exact, make_optimizer, with_correlated_events
from cluster_expansion import ClusterExpansion, dem_supernode_groups
from dem_viz import BLUE, ORANGE, AQUA, GRAY

INK = "#2b2b2b"


# ---------------------------------------------------------------- builders --

def surface_dem(distance=3, rounds=None, p=0.01, circuit_level=False):
    """stim circuit + DEM + DemTN for a rotated memory_z surface code.

    circuit_level=False gives the phenomenological model (data depolarization
    before each round + measurement flips); True adds after-Clifford
    depolarization and reset flips. Hyperedges are kept
    (decompose_errors=False) in both cases."""
    rounds = distance if rounds is None else rounds
    noise = dict(before_round_data_depolarization=p,
                 before_measure_flip_probability=p)
    if circuit_level:
        noise.update(after_clifford_depolarization=p,
                     after_reset_flip_probability=p)
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=rounds,
        **noise)
    dem = circuit.detector_error_model(decompose_errors=False,
                                       flatten_loops=True)
    return circuit, dem, DemTN(dem)


def bb_phenom_dem_text(H, p, rounds):
    """Phenomenological DEM text for a check matrix H: a data flip per qubit
    per round (fires that round's comparisons of its checks) and a
    measurement flip per check for all but the last round (fires two
    consecutive comparisons). Detector index = round * num_checks + check."""
    m, n = H.shape
    lines = []
    for t in range(rounds):
        for j in range(n):
            dets = " ".join(f"D{t * m + c}" for c in np.where(H[:, j])[0])
            lines.append(f"error({p}) {dets}")
    for t in range(rounds - 1):
        for c in range(m):
            lines.append(f"error({p}) D{t * m + c} D{(t + 1) * m + c}")
    return "\n".join(lines)


def bb_phenom_dem(p=0.01, rounds=3):
    """(H, dem, DemTN) for the BB [[72,12,6]] code under phenomenological
    noise. Needs the quits package (same dependency as qlpdc_bp_loops)."""
    from quits.qldpc_code.bb import BbCode
    H = BbCode(6, 6, [3], [1, 2], [1, 2], [3]).hz.astype(np.uint8)
    dem = stim.DetectorErrorModel.from_file(
        io.StringIO(bb_phenom_dem_text(H, p, rounds)))
    return H, dem, DemTN(dem)


def build_closed(dt, syndrome):
    """dt.build with every open observable leg capped by the all-ones vector,
    so the contraction returns the likelihood Pr(s) (logical classes
    marginalized)."""
    tn = dt.build(syndrome, observables=None)
    for ix in list(tn.outer_inds()):
        tn |= qtn.Tensor(np.ones(2), inds=(ix,), tags="obscap")
    return tn


def sample_syndromes(circuit, n, seed=7, nonzero=True, oversample=None):
    """First n detector samples from the circuit (nonzero ones by default)."""
    shots = circuit.compile_detector_sampler(seed=seed).sample(
        shots=oversample or max(4 * n, 64))
    out = [s.astype(np.uint8) for s in shots if (s.sum() > 0 or not nonzero)]
    return out[:n]


# ------------------------------------------------- correlated-noise twins --

def inject_correlated_pairs(dem, dt, n_pairs=4, pc=0.005):
    """Marginal-matched twin construction. Picks n_pairs pairs of two-detector
    mechanisms sharing one detector; each injected event fires the symmetric
    difference of the pair's detectors jointly at probability pc. The
    independent twin gets the same detectors as separate singles, so every
    single-detector rate is identical between the two models and only the
    correlations differ. Returns a dict with both DEMs, both DemTNs, the
    detector pairs, and the number of base mechanisms (injected events have
    index >= n_base)."""
    pairs, used = [], set()
    for i in range(dt.num_events):
        if len(dt.event_dets[i]) != 2:
            continue
        for k in range(i + 1, dt.num_events):
            if len(dt.event_dets[k]) != 2:
                continue
            a, b = set(dt.event_dets[i]), set(dt.event_dets[k])
            if len(a & b) == 1:
                ds = tuple(sorted(a ^ b))
                if not (set(ds) & used):
                    pairs.append(ds)
                    used |= set(ds)
            if len(pairs) >= n_pairs:
                break
        if len(pairs) >= n_pairs:
            break
    dem_corr, dem_indep = with_correlated_events(
        dem, [(pc, list(ds)) for ds in pairs])
    return dict(pairs=pairs, n_base=dt.num_events,
                dem_corr=dem_corr, dem_indep=dem_indep,
                dt_corr=DemTN(dem_corr), dt_indep=DemTN(dem_indep))


def _ce_loops(twins, circuit_level, M):
    """Loop sets for the cluster expansion of each twin. Phenomenological
    networks are loop-sparse, so the full edge-metric enumeration to weight M
    is used. Circuit-level networks are not (bulk detectors are ~50-way
    hubs), so loops are restricted to those touching the injected events;
    everything else cancels in the likelihood ratio."""
    loops = {}
    for key in ("corr", "indep"):
        dt = twins[f"dt_{key}"]
        ce = ClusterExpansion(build_closed(dt, np.zeros(dt.E.shape[0],
                                                        dtype=np.uint8)))
        if circuit_level:
            groups = dem_supernode_groups(ce.tn)
            touch = {f"copy{i}" for i in range(twins["n_base"], dt.num_events)}
            loops[key] = ce.gen_loops_grouped_touching(M, groups, touch)
        else:
            loops[key] = ce.gen_loops(M)
    return loops


def correlated_llr(twins, n_shots=100, circuit_level=False, M=None,
                   seed=42, exact=True, progress=25):
    """Per-shot LLR between the twins, evaluated three ways.

    Returns an (n_shots, 3) array: columns are the exact, BP-only, and
    BP+cluster estimates of ln P_corr(x) - ln P_indep(x) for syndromes x
    sampled from the correlated model. Set exact=False when the network is
    too wide to contract (the exact column is then NaN). M defaults to 6
    (phenomenological, global loops) or 4 (circuit level, local loops)."""
    M = M or (4 if circuit_level else 6)
    loops = _ce_loops(twins, circuit_level, M)
    opt = make_optimizer(max_repeats=16)
    if exact:  # warm the contraction trees once
        for key in ("corr", "indep"):
            dt = twins[f"dt_{key}"]
            contract_exact(build_closed(dt, np.zeros(dt.E.shape[0],
                                                     dtype=np.uint8)),
                           optimize=opt)
    dets = np.asarray(
        twins["dem_corr"].compile_sampler(seed=seed).sample(shots=n_shots)[0]
    ).astype(np.uint8)

    rows = []
    for k, syn in enumerate(dets):
        vals = {}
        for key in ("corr", "indep"):
            dt = twins[f"dt_{key}"]
            tn = build_closed(dt, syn)
            lex = (float(np.log(contract_exact(tn, optimize=opt)))
                   if exact else np.nan)
            ce = ClusterExpansion(tn)
            lbp = float(np.log(abs(ce.contract_bp())))
            if circuit_level:
                lce = float(np.log(abs(
                    ce.contract_grouped(M, loops_w=loops[key]))))
            else:
                ce.gen_loops = lambda mw, K=key: [F for F in loops[K]
                                                 if len(F) <= mw]
                lce = float(np.log(abs(ce.contract(M))))
            vals[key] = (lex, lbp, lce)
        rows.append(tuple(vals["corr"][j] - vals["indep"][j]
                          for j in range(3)))
        if progress and (k + 1) % progress == 0:
            print(f"  {k + 1}/{n_shots} shots")
    return np.array(rows)


def loop_spectra(twins, n_shots=10, circuit_level=False, M=None, seed=42):
    """Mean |z_l| per loop in each twin, plus a predicate marking loops that
    pass through the injected events (those have no counterpart in the
    independent model). Feeds fig_fingerprints."""
    M = M or (4 if circuit_level else 6)
    loops = _ce_loops(twins, circuit_level, M)
    if circuit_level:
        loops = {k: [F for F, w in v] for k, v in loops.items()}
    dets = np.asarray(
        twins["dem_corr"].compile_sampler(seed=seed).sample(shots=n_shots)[0]
    ).astype(np.uint8)
    acc = {"corr": {}, "indep": {}}
    for syn in dets:
        for key in ("corr", "indep"):
            ce = ClusterExpansion(build_closed(twins[f"dt_{key}"], syn))
            for F in loops[key]:
                acc[key].setdefault(frozenset(F), []).append(
                    abs(ce.loop_correction(F)))
    mean_c = {k: np.mean(v) for k, v in acc["corr"].items()}
    mean_d = {k: np.mean(v) for k, v in acc["indep"].items()}
    n_base = twins["n_base"]

    def is_new(F):
        for ix in F:
            if ix.startswith("e") and "_" in ix:
                try:
                    if int(ix[1:ix.index("_")]) >= n_base:
                        return True
                except ValueError:
                    pass
        return False

    return mean_c, mean_d, is_new


# ------------------------------------------------------------------ figures --

def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_llr_combined(rows_phenom, rows_circuit, path=None, sprt_shots=300):
    """Top: sequential (SPRT) evidence accumulation on the phenomenological
    rows. Bottom: circuit-level per-shot LLR against exact. This is the
    slides' fig24."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2))

    ax = axes[0]
    n = min(sprt_shots, len(rows_phenom))
    for lev, lab in ((np.log(10), r"$\alpha=10^{-1}$"),
                     (np.log(100), r"$\alpha=10^{-2}$"),
                     (np.log(1000), r"$\alpha=10^{-3}$")):
        ax.axhline(lev, color=BLUE, ls=":", lw=1.0, alpha=0.7)
        ax.annotate(lab, (n * 1.005, lev), fontsize=8, color=BLUE,
                    va="center", annotation_clip=False)
    for lab, v, col in (("exact", rows_phenom[:, 0], INK),
                        ("BP + clusters", rows_phenom[:, 2], ORANGE),
                        ("BP only", rows_phenom[:, 1], GRAY)):
        ax.plot(np.cumsum(v)[:n], color=col, lw=1.6, label=lab)
    ax.set_ylabel(r"$\Lambda_N$ (summed LLR difference)")
    ax.set_xlabel("shots collected")
    ax.set_title("PHENOMENOLOGICAL, d=3: sequential test for correlated noise",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, n)

    ax = axes[1]
    ex, bp, ce = rows_circuit[:, 0], rows_circuit[:, 1], rows_circuit[:, 2]
    o = np.argsort(ex)
    xs = np.arange(len(ex))
    ax.plot(xs, ex[o], "-", color=INK, lw=1.8, label="exact", zorder=3)
    ax.plot(xs, bp[o], ".", color=GRAY, ms=6,
            label=f"BP only (mean |err| {np.abs(bp - ex).mean():.4f})")
    ax.plot(xs, ce[o], ".", color=ORANGE, ms=6,
            label=f"BP + local clusters (mean |err| {np.abs(ce - ex).mean():.4f})")
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.set_xlabel("shot (sorted by exact LLR)")
    ax.set_ylabel("per-shot LLR")
    ax.set_title("CIRCUIT LEVEL, d=3: per-shot LLR against exact",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        _clean(ax)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# Convergence data for the d=5 phenomenological error-vs-order panel:
# |log Pr - compressed chi=8| from the seeded d=5 study (sampler seed 7).
# Recomputing it takes ~15 min of compressed sweeps, so the measured values
# are recorded here with their provenance.
D5_CONVERGENCE = {
    0: (7.16e-6, [6.77e-6, 2.44e-6, 1.73e-6, 1.56e-6, 1.50e-6]),
    7: (2.15e-4, [2.07e-4, 1.50e-4, 1.49e-4, 1.49e-4, 1.60e-4]),
}


def fig_fingerprints(spec_phenom, spec_circuit, path=None, top=15,
                     d5_convergence=D5_CONVERGENCE):
    """Loop-correction spectra of the twins (top, phenomenological and
    circuit level side by side) over the d=5 error-vs-order convergence
    panel. This is the slides' fig25."""
    fig = plt.figure(figsize=(10.5, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.42,
                          wspace=0.18)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_conv = fig.add_subplot(gs[1, :])

    for ax, (mc, md, is_new), title in (
            (axes[0], spec_phenom, "PHENOMENOLOGICAL, d=3"),
            (axes[1], spec_circuit, "CIRCUIT LEVEL, d=3 (local loops)")):
        keys = sorted(mc, key=lambda k: -mc[k])[:top]
        for x, k in enumerate(keys):
            ax.bar(x, mc[k], width=0.8, color=ORANGE if is_new(k) else GRAY,
                   alpha=0.95 if is_new(k) else 0.6)
            if k in md:
                ax.plot(x, md[k], "_", color=INK, ms=9, mew=1.4)
        ax.set_yscale("log")
        ax.set_xlabel("loop rank")
        ax.set_xticks(range(0, top, 2))
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        _clean(ax)
    axes[0].set_ylabel(r"$|z_l|$ (mean over shots)")
    from matplotlib.lines import Line2D
    axes[1].legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE),
                 plt.Rectangle((0, 0), 1, 1, color=GRAY, alpha=0.6),
                 Line2D([0], [0], marker="_", color=INK, lw=0, ms=9, mew=1.4)],
        labels=["loops from correlated events", "loops in both models",
                "value in independent model"],
        frameon=False, fontsize=8)

    MS = [4, 5, 6, 7, 8]
    for k, (wt, (ebp, errs)) in enumerate(d5_convergence.items()):
        ax_conv.plot([3] + MS, [ebp] + errs, "o-", color=BLUE, lw=1.6, ms=4.5,
                     alpha=0.45 + 0.22 * k, label=f"shot wt {wt}")
    ax_conv.set_yscale("log")
    ax_conv.set_xticks([3] + MS, ["BP"] + [f"M={m}" for m in MS])
    ax_conv.set_ylabel(r"|logPr error| vs compressed $\chi$=8")
    ax_conv.set_title("PHENOMENOLOGICAL, d=5: error falls as loops are added, "
                      r"until $\chi=8$'s own accuracy is reached",
                      fontsize=10, color=INK, loc="left")
    ax_conv.legend(frameon=False, fontsize=8)
    _clean(ax_conv)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def fig_repcode_network(path=None, p=0.01, syndrome_bits=(2, 3)):
    """The two-round d=3 repetition-code network with every tensor's entries
    printed (matrices read from the constructed network). This is the slides'
    fig12; syndrome_bits are the fired detectors of the displayed shot."""
    FS = 6.0
    WB = dict(boxstyle="square,pad=0.16", facecolor="white",
              edgecolor="none", alpha=0.92)
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=2,
        before_round_data_depolarization=p, before_measure_flip_probability=p)
    dem = circuit.detector_error_model(flatten_loops=True)
    dt = DemTN(dem)
    coords = {k: (v[0], v[1])
              for k, v in circuit.get_detector_coordinates().items()}
    syn = np.zeros(dem.num_detectors, dtype=np.uint8)
    for b in syndrome_bits:
        syn[b] = 1
    tn = dt.build(syn, observables=None)

    def tensor_of(tag):
        (tid,) = tn.tag_map[tag]
        return tn.tensor_map[tid]

    def mat_str(arr, float_fmt=False):
        a = np.asarray(arr)
        m = a.reshape(2, -1) if a.ndim > 1 else a.reshape(1, -1)
        lines = [" ".join(f"{x:.3f}"[1:] if float_fmt else f"{x:.0f}"
                          for x in row) for row in m]
        if len(lines) == 1:
            return "[" + lines[0] + "]"
        return "⎡" + lines[0] + "⎤\n⎣" + lines[1] + "⎦"

    det_pos = {d: (x, t) for d, (x, t) in coords.items()}
    mech_pos, bern_off = {}, {}
    for i, dets in enumerate(dt.event_dets):
        obs = dt.event_obs[i]
        pts = [det_pos[d] for d in dets]
        if len(dets) == 2 and pts[0][1] == pts[1][1]:
            mech_pos[i] = ((pts[0][0] + pts[1][0]) / 2, pts[0][1])
            bern_off[i] = (0, -0.24)
        elif len(dets) == 2:
            x = pts[0][0]
            mech_pos[i] = (x, (pts[0][1] + pts[1][1]) / 2)
            bern_off[i] = (-0.34, 0) if x == 1 else (0.34, 0)
        elif obs:
            mech_pos[i] = (pts[0][0] + 1, pts[0][1])
            bern_off[i] = (0, -0.24)
        else:
            mech_pos[i] = (pts[0][0] - 1, pts[0][1])
            bern_off[i] = (0, -0.24)
    obs_pos = (5.05, 1.0)

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    for i, dets in enumerate(dt.event_dets):
        mx, my = mech_pos[i]
        for d in dets:
            ax.plot([mx, det_pos[d][0]], [my, det_pos[d][1]], color=GRAY,
                    lw=1.4, zorder=1)
        if dt.event_obs[i]:
            ax.plot([mx, obs_pos[0]], [my, obs_pos[1]], color=GRAY, lw=1.4,
                    zorder=1)
        bx, by = mx + bern_off[i][0], my + bern_off[i][1]
        ax.plot([mx, bx], [my, by], color=GRAY, lw=1.4, zorder=1)
    ax.plot([obs_pos[0], obs_pos[0] + 0.5], [obs_pos[1], obs_pos[1]],
            color=INK, lw=1.5, ls=(0, (3, 2)), zorder=1)
    ax.annotate("open leg\n(logical class)", (obs_pos[0] + 0.55, obs_pos[1]),
                fontsize=9, color=INK, va="center")
    for d, (x, t) in det_pos.items():
        fired = bool(syn[d])
        ax.scatter([x], [t], marker="s", s=520, c=BLUE, zorder=3,
                   edgecolors=ORANGE if fired else "white",
                   linewidths=2.8 if fired else 1.1)
        ax.text(x, t, r"$\oplus$", ha="center", va="center", fontsize=13,
                color="white", zorder=4)
        ax.annotate(mat_str(tensor_of(f"det{d}").data), (x, t - 0.155),
                    fontsize=FS, family="monospace", color=INK, ha="center",
                    va="top", zorder=2.5, bbox=WB)
    for i in range(dt.num_events):
        mx, my = mech_pos[i]
        ax.scatter([mx], [my], marker="o", s=360, c=ORANGE, zorder=3,
                   edgecolors="white", linewidths=1.1)
        ax.text(mx, my, r"$\delta$", ha="center", va="center", fontsize=11,
                color="white", zorder=4)
        s = mat_str(tensor_of(f"copy{i}").data)
        if bern_off[i][0] != 0:
            ha = "left" if bern_off[i][0] < 0 else "right"
            ax.annotate(s, (mx + (0.12 if ha == "left" else -0.12), my),
                        fontsize=FS, family="monospace", color=INK, ha=ha,
                        va="center", zorder=2.5, bbox=WB)
        else:
            ax.annotate(s, (mx, my + 0.13), fontsize=FS, family="monospace",
                        color=INK, ha="center", va="bottom", zorder=2.5,
                        bbox=WB)
        bx, by = mx + bern_off[i][0], my + bern_off[i][1]
        ax.scatter([bx], [by], marker="D", s=150, c=AQUA, zorder=3,
                   edgecolors="white", linewidths=0.9)
        s = mat_str(tensor_of(f"event{i}").data, float_fmt=True)
        if bern_off[i][0] != 0:
            ha = "right" if bern_off[i][0] < 0 else "left"
            ax.annotate(s, (bx - 0.08 if ha == "right" else bx + 0.08, by),
                        fontsize=FS, family="monospace", color="#0e7a54",
                        ha=ha, va="center", zorder=2.5, bbox=WB)
        else:
            ax.annotate(s, (bx, by - 0.10), fontsize=FS, family="monospace",
                        color="#0e7a54", ha="center", va="top", zorder=2.5,
                        bbox=WB)
    ax.scatter([obs_pos[0]], [obs_pos[1]], marker="s", s=520, c=BLUE,
               zorder=3, edgecolors="white", linewidths=1.1)
    ax.text(*obs_pos, r"$\oplus$", ha="center", va="center", fontsize=13,
            color="white", zorder=4)
    ax.annotate(mat_str(tensor_of("obs0").data),
                (obs_pos[0], obs_pos[1] - 0.155), fontsize=FS,
                family="monospace", color=INK, ha="center", va="top",
                zorder=2.5, bbox=WB)
    for t in (0, 1, 2):
        ax.annotate(f"round {t}", (-1.3, t), fontsize=10, color="#7a7974",
                    va="center")
    ax.annotate("", xy=(-1.2, 2.4), xytext=(-1.2, -0.35),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.3))
    legend = [("s", BLUE, 170, r"XOR tensor, clamped to detector bit $x_d$"),
              ("o", ORANGE, 130, "COPY tensor of one error mechanism"),
              ("D", AQUA, 75, r"Bernoulli prior $[1-p_i,\ p_i]$")]
    handles = [plt.scatter([], [], marker=m, s=s, c=c, edgecolors="white")
               for m, c, s, lab in legend]
    ax.legend(handles, [lab for _, _, _, lab in legend], frameon=False,
              fontsize=10.5, loc="upper left", bbox_to_anchor=(-0.02, 1.03))
    ax.set_xlim(-1.55, 6.7)
    ax.set_ylim(-0.75, 2.9)
    ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight")
    return fig


def fig_dem_zoo(path=None, p_circuit=0.003, p_bb=0.01, bb_rounds=3):
    """d=5 circuit-level surface DEM (3D spacetime) over the BB [[72,12,6]]
    phenomenological DEM (nonlocal), same node legend as the repcode figure.
    This is the slides' fig26. Needs quits for the BB half."""
    fig = plt.figure(figsize=(9.5, 10.5))

    _, dem5, dt5 = surface_dem(5, 5, p_circuit, circuit_level=True)
    c5 = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=5, rounds=5,
        after_clifford_depolarization=p_circuit,
        before_measure_flip_probability=p_circuit,
        after_reset_flip_probability=p_circuit,
        before_round_data_depolarization=p_circuit)
    dxyz = {d: tuple(v) for d, v in c5.get_detector_coordinates().items()}
    ax = fig.add_subplot(2, 1, 1, projection="3d")
    for dets in dt5.event_dets:
        if not dets:
            continue
        pts = np.array([dxyz[d] for d in dets])
        cen = pts.mean(0)
        for p_ in pts:
            ax.plot(*zip(cen, p_), color=GRAY, lw=0.25, alpha=0.25)
    cens = np.array([np.mean([dxyz[d] for d in dets], axis=0)
                     for dets in dt5.event_dets if dets])
    ax.scatter(cens[:, 0], cens[:, 1], cens[:, 2], s=3, c=ORANGE, alpha=0.7,
               depthshade=False)
    ax.scatter(cens[:, 0], cens[:, 1], cens[:, 2] - 0.10, s=1.2, c=AQUA,
               alpha=0.5, marker="D", depthshade=False)
    dp = np.array(list(dxyz.values()))
    ax.scatter(dp[:, 0], dp[:, 1], dp[:, 2], s=26, c=BLUE, marker="s",
               depthshade=False, zorder=5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("round")
    ax.set_title(f"SURFACE $d=5$, CIRCUIT-LEVEL NOISE: {dt5.num_events} "
                 f"mechanisms, {dem5.num_detectors} detectors "
                 "(3D spacetime DEM)", fontsize=10, color=INK, loc="left")
    ax.view_init(elev=16, azim=-62)
    legend = [("s", BLUE, 90, r"XOR tensor, clamped to detector bit $x_d$"),
              ("o", ORANGE, 60, "COPY tensor of one error mechanism"),
              ("D", AQUA, 35, r"Bernoulli prior $[1-p_i,\ p_i]$")]
    handles = [plt.scatter([], [], marker=m, s=s, c=col, edgecolors="white")
               for m, col, s, lab in legend]
    ax.legend(handles, [lab for _, _, _, lab in legend], frameon=False,
              fontsize=9, loc="upper left", bbox_to_anchor=(-0.05, 1.0))

    H, demb, dtb = bb_phenom_dem(p_bb, bb_rounds)
    m = H.shape[0]
    ax = fig.add_subplot(2, 1, 2)
    det_pos = {t * m + c: (c, 2.2 * t)
               for t in range(bb_rounds) for c in range(m)}
    for dets in dtb.event_dets:
        pts = [det_pos[d] for d in dets]
        if len(dets) == 2 and abs(pts[0][1] - pts[1][1]) > 1:
            cx, cy = pts[0][0], (pts[0][1] + pts[1][1]) / 2
        else:
            cx = np.mean([p_[0] for p_ in pts])
            cy = pts[0][1] - 0.75
        for p_ in pts:
            ax.plot([cx, p_[0]], [cy, p_[1]], color=GRAY, lw=0.3, alpha=0.3,
                    zorder=1)
        ax.scatter([cx], [cy], s=8, c=ORANGE, zorder=3)
        ax.scatter([cx], [cy - 0.16], s=3.5, c=AQUA, marker="D", zorder=3)
    dp = np.array(list(det_pos.values()))
    ax.scatter(dp[:, 0], dp[:, 1], s=30, c=BLUE, marker="s", zorder=4)
    for t in range(bb_rounds):
        ax.annotate(f"round {t}", (-3.3, 2.2 * t), fontsize=9,
                    color="#7a7974", va="center")
    ax.set_title(f"BB $[[72,12,6]]$, PHENOMENOLOGICAL NOISE, {bb_rounds} "
                 f"rounds: {dtb.num_events} mechanisms, {demb.num_detectors} "
                 "detectors (nonlocal expander DEM)", fontsize=10, color=INK,
                 loc="left")
    ax.set_xlim(-4, m + 0.5)
    ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


# Wall times per likelihood evaluation measured in the phenomenological
# campaign (p=0.01, this machine); None marks a method that cannot run.
COST_REACH_TIMES = {
    "surface d=3\n(width 14)": {"exact": 0.05, "compressed": 0.4, "ce": 0.3},
    "surface d=5\n(width 45)": {"exact": None, "compressed": 37.0, "ce": 2.0},
    "BB [[72,12,6]]\n(expander)": {"exact": None, "compressed": None,
                                   "ce": 2.0},
}


def fig_cost_reach(path=None, times=COST_REACH_TIMES):
    """Cost per likelihood evaluation across code families; hatched bars mark
    methods that cannot run at all. This is the slides' fig17."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    methods = [("exact", INK), ("compressed $\\chi$=8", BLUE),
               ("BP + clusters", ORANGE)]
    keys = ["exact", "compressed", "ce"]
    wd = 0.26
    for j, ((mname, col), key) in enumerate(zip(methods, keys)):
        for i, (code, tt) in enumerate(times.items()):
            t = tt[key]
            xpos = i + (j - 1) * wd
            if t is None:
                ax.bar(xpos, 600, width=wd, color="none", edgecolor=col,
                       hatch="///", lw=1.0)
                lab = ("infeasible\n($2^{45}$ memory)" if key == "exact"
                       else "no valid\ngeometry")
                ax.annotate(lab, (xpos, 700), ha="center", fontsize=7,
                            color=col)
            else:
                ax.bar(xpos, t, width=wd, color=col)
                ax.annotate(f"{t:g}s", (xpos, t * 1.15), ha="center",
                            fontsize=7.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in methods]
    ax.legend(handles, [m for m, _ in methods], frameon=False, fontsize=8.5,
              loc="upper left")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 3000)
    ax.set_xticks(range(len(times)), list(times.keys()), fontsize=8.5)
    ax.set_ylabel("wall time per syndrome (s)")
    ax.set_title("phenomenological noise, p=0.01: cost per likelihood "
                 "evaluation", fontsize=10.5, color=INK)
    _clean(ax)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
