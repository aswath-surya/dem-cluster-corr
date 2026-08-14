"""Clean matplotlib figures for the DEM tensor-network / cluster-expansion
results. One figure = one message; series colors follow the entity across
figures (blue = cluster expansion / correlated model, orange = BP / decorrelated
model, gray = context), text stays in ink colors, grids are recessive.
"""

import numpy as np
import matplotlib.pyplot as plt

# palette (validated categorical slots, light mode) + ink/surface tokens
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#b0afa9"          # de-emphasis / context marks
INK = "#0b0b0b"           # primary text
INK2 = "#52514e"          # secondary text
GRID = "#eceae6"          # one step off surface
SURFACE = "#fcfcfb"

_MARK = dict(markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.6)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _titles(fig, title, subtitle):
    fig.text(0.02, 0.975, title, fontsize=12, fontweight="bold",
             color=INK, va="top", ha="left")
    if subtitle:
        fig.text(0.02, 0.925, subtitle, fontsize=9.5, color=INK2,
                 va="top", ha="left")


def _new_fig(width=6.4, height=4.2, title="", subtitle=""):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    _style(ax)
    _titles(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig, ax


def plot_convergence(curves):
    """Relative error of Pr(x) vs cluster truncation weight m.

    curves: dict {label: (list_of_m, list_of_rel_errs)}; m=0 is plain BP.
    """
    fig, ax = _new_fig(
        title="Cluster expansion converges exponentially in loop weight",
        subtitle="relative error of Pr(x) vs truncation weight m  (m = 0 is plain BP)",
    )
    colors = [BLUE, AQUA, ORANGE]
    for (label, (ms, errs)), c in zip(curves.items(), colors):
        ax.plot(ms, errs, "-o", color=c, linewidth=2, label=label, **_MARK)
        ax.annotate(label, (ms[-1], errs[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=9, color=INK2,
                    va="center")
    ax.set_yscale("log")
    ax.set_xlabel("cluster weight m", fontsize=10, color=INK2)
    ax.set_ylabel("relative error", fontsize=10, color=INK2)
    ax.set_xticks(sorted({m for _, (ms, _) in curves.items() for m in ms}))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    ax.margins(x=0.22)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig


def plot_loop_decay(ps, allowed, forbidden,
                    forbidden_label="|l| = 6  (identically zero)"):
    """DEM loop decay vs physical error rate.

    ps: error rates; allowed: max|Z_l| at |l|=8 per p; forbidden: max|Z_l|
    over the identically-vanishing classes per p (|l|=6 loops only exist via
    the marginalized observable node, whose rank-1 all-ones tensor supports
    no excitation; genuine triangles are forbidden by the bipartite grid).
    """
    fig, ax = _new_fig(
        title="Loop corrections grow with noise rate; |l| = 6 vanishes exactly",
        subtitle="max |Zₗ| per loop-weight class, d=3 R=3 repetition-code DEM",
    )
    ax.plot(ps, allowed, "-o", color=BLUE, linewidth=2,
            label="|l| = 8  (allowed)", **_MARK)
    ax.plot(ps, forbidden, "-o", color=GRAY, linewidth=2,
            label=forbidden_label, **_MARK)
    ax.annotate("carries the whole expansion", (ps[-1], allowed[-1]),
                xytext=(-4, 12), textcoords="offset points", fontsize=9,
                color=INK2, ha="right")
    ax.annotate("numerical zero", (ps[-1], forbidden[-1]),
                xytext=(-4, 12), textcoords="offset points", fontsize=9,
                color=INK2, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate p", fontsize=10, color=INK2)
    ax.set_ylabel("max |Zₗ|", fontsize=10, color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="center left")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig


def plot_correlation_spectrum(spec, floor=1e-40):
    """Cleveland dot plot: which loops carry the injected correlation.

    spec: list of (row_label, corr_value, decorr_value); values <= floor (or
    None) are drawn as open markers at the floor ("absent / exactly zero").
    """
    fig, ax = _new_fig(
        width=7.0, height=3.6,
        title="The injected correlation lives in specific loops",
        subtitle="max |Zₗ| by loop class, correlated vs decorrelated DEM "
                 "(matched single-detector marginals)",
    )
    ys = np.arange(len(spec))[::-1]
    dodge = {"corr": +0.16, "decorr": -0.16}
    for y, (label, vc, vd) in zip(ys, spec):
        for val, color, name in ((vc, BLUE, "corr"), (vd, ORANGE, "decorr")):
            v = floor if (val is None or val <= floor) else val
            absent = val is None or val <= floor
            ax.plot([v], [y + dodge[name]], "o",
                    color=SURFACE if absent else color,
                    markeredgecolor=color,
                    markeredgewidth=1.6,
                    markersize=8, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in spec], fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("max |Zₗ|   (open marker = absent / exactly zero)",
                  fontsize=10, color=INK2)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.yaxis.grid(False)
    ax.set_ylim(-0.6, len(spec) - 0.4)
    handles = [
        plt.Line2D([], [], marker="o", color=BLUE, linestyle="", **_MARK,
                   label="correlated DEM"),
        plt.Line2D([], [], marker="o", color=ORANGE, linestyle="", **_MARK,
                   label="decorrelated DEM"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK2,
              loc="lower left", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig


def plot_llr(rows, shot_weights=None):
    """Per-shot log-likelihood-ratio recovery.

    rows: (n_shots, 3) array of LLR values, columns (exact, BP, cluster m=8).
    Top: LLR per shot (exact = ink tick, BP = orange, cluster = blue).
    Bottom: |LLR error| vs exact, log scale.
    """
    rows = np.asarray(rows)
    n = len(rows)
    x = np.arange(n)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.8, 6.0), facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1], "hspace": 0.18},
    )
    _style(ax1)
    _style(ax2)
    _titles(
        fig,
        "BP under-reads the correlation; clusters recover it",
        f"per-shot LLR = log Pr(corr) − log Pr(decorr);  "
        f"mean exact LLR = {rows[:, 0].mean():+.3f} nats/shot",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    ax1.axhline(0, color=GRID, linewidth=1)
    ax1.plot(x, rows[:, 0], "_", color=INK, markersize=15, markeredgewidth=2.2,
             label="exact", zorder=4)
    ax1.plot(x, rows[:, 1], "o", color=ORANGE, **_MARK, linestyle="", label="BP",
             zorder=3)
    ax1.plot(x, rows[:, 2], "o", color=BLUE, **_MARK, linestyle="",
             label="cluster m=8", zorder=3)
    ax1.set_ylabel("LLR (nats)", fontsize=10, color=INK2)
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=3,
               loc="upper right")

    err_bp = np.abs(rows[:, 1] - rows[:, 0])
    err_cl = np.abs(rows[:, 2] - rows[:, 0])
    ax2.plot(x, err_bp, "o", color=ORANGE, **_MARK, linestyle="", label="BP")
    ax2.plot(x, err_cl, "o", color=BLUE, **_MARK, linestyle="",
             label="cluster m=8")
    ax2.set_yscale("log")
    ax2.set_ylabel("|LLR error|", fontsize=10, color=INK2)
    ax2.set_xlabel("shot", fontsize=10, color=INK2)
    ax2.set_xticks(x)
    if shot_weights is not None:
        ax2.set_xticklabels([f"{i}\n(w{w})" for i, w in zip(x, shot_weights)],
                            fontsize=8)
    worst = int(np.argmax(err_bp))
    ax2.annotate("syndrome excites the\ncorrelated detectors",
                 (worst, err_bp[worst]), xytext=(-12, -4),
                 textcoords="offset points", fontsize=8.5, color=INK2,
                 ha="right", va="top")
    return fig


def draw_dem_tn(demtn, tn, dem=None, corr_ids=None, figsize=(9, 6),
                iterations=200, node_scale=1.6, show_tags=True):
    """Draw the actual tensor network with quimb, one color per tensor role.

    demtn: the DemTN whose `build` produced `tn`; dem: the stim DEM, used to
    pin detector parity tensors to their (space, time) coordinates so the
    layout is physical rather than a spring-embedding hairball; corr_ids:
    event indices to highlight (e.g. injected correlated mechanisms). The
    point of the figure: an error mechanism is not an edge of the network,
    it is a NODE (Bernoulli vector feeding a COPY tensor), wired by one edge
    to every detector parity node it can flip. A correlated event is
    therefore an extra node with several detector edges, which is exactly
    what lets one hidden variable feed many detectors at once.
    """
    tnd = tn.copy()
    corr_ids = set(corr_ids or ())
    role_of = {}
    for tid, t in tnd.tensor_map.items():
        tags = t.tags
        role = None
        for tag in tags:
            if tag.startswith("event"):
                i = int(tag[5:])
                role = "injected event" if i in corr_ids else "mechanism (Bernoulli)"
            elif tag.startswith("copy"):
                i = int(tag[4:])
                role = "injected event" if i in corr_ids else "broadcast (COPY)"
            elif tag.startswith("det"):
                role = "detector parity (XOR)"
            elif tag.startswith("obs"):
                role = "observable parity"
            if role:
                break
        role_of[tid] = role or "other"
        t.add_tag(role_of[tid])

    fix = {}
    if dem is not None:
        coords = dem.get_detector_coordinates()
        for tid, t in tnd.tensor_map.items():
            for tag in t.tags:
                if tag.startswith("det") and tag[3:].isdigit():
                    d = int(tag[3:])
                    if d in coords:
                        c = coords[d]
                        fix[tid] = (float(c[0]),
                                    float(c[1]) if len(c) > 1 else 0.0)

    highlight = [tid for tid, r in role_of.items() if r == "injected event"]
    fig = tnd.draw(
        color=["mechanism (Bernoulli)", "broadcast (COPY)",
               "detector parity (XOR)", "observable parity", "injected event"],
        custom_colors=[GRAY, AQUA, BLUE, "#4a3aa7", ORANGE],
        fix=fix or None,
        highlight_tids=highlight,
        highlight_tids_color=ORANGE,
        legend=True,
        iterations=iterations,
        node_scale=node_scale,
        show_tags=show_tags,
        figsize=figsize,
        return_fig=True,
    )
    return fig


def plot_backend_timing(rows):
    """Exact-contraction wall time per backend across surface-code DEM sizes.

    rows: list of (label, width, t_numpy_ms, t_torchcpu_ms, t_mps_ms_or_None).
    A None MPS entry means the contraction is impossible on Metal (tensor
    rank cap of 16); it is drawn as an open marker at the top of the axis.
    """
    fig, ax = _new_fig(
        width=7.0, height=4.4,
        title="Exact contraction: CPU wins at these sizes; Apple GPU cannot run them",
        subtitle="wall time per backend, same contraction path everywhere; "
                 "open marker = fails on Metal's 16-dim tensor rank cap",
    )
    x = np.arange(len(rows))
    labels = [r[0] for r in rows]
    t_np = [r[2] for r in rows]
    t_tc = [r[3] for r in rows]
    ax.plot(x, t_np, "-o", color=BLUE, linewidth=2, label="numpy float64 (CPU)",
            **_MARK)
    ax.plot(x, t_tc, "-o", color=AQUA, linewidth=2, label="torch float64 (CPU)",
            **_MARK)
    mps_x = [i for i, r in enumerate(rows) if r[4] is not None]
    mps_t = [rows[i][4] for i in mps_x]
    fail_x = [i for i, r in enumerate(rows) if r[4] is None]
    ax.plot(mps_x, mps_t, "o", color=ORANGE, linestyle="",
            label="torch float32 (MPS / GPU)", **_MARK)
    top = max(max(t_np), max(t_tc)) * 3
    ax.plot(fail_x, [top] * len(fail_x), "o", color=SURFACE,
            markeredgecolor=ORANGE, markeredgewidth=1.6, markersize=8)
    if fail_x:
        ax.annotate("MPS fails: rank > 16", (fail_x[0], top), xytext=(0, 8),
                    textcoords="offset points", fontsize=9, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("contraction time (ms)", fontsize=10, color=INK2)
    ax.set_xlabel("surface-code DEM (distance, rounds, contraction width)",
                  fontsize=10, color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ax.margins(y=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig


def plot_loop_class_comparison(rep_counts, surf_counts):
    """Nonzero loop fraction per weight class: repetition vs surface DEM.

    rep_counts / surf_counts: dict {weight: (n_nonzero, n_total)}. Shows the
    structural difference hyperedges make: the repetition code's bipartite
    detector graph forbids odd event-cycles so its |l|=6 class is empty or
    identically zero, while depolarizing hyperedges in the surface code
    create genuine short loops.
    """
    fig, ax = _new_fig(
        width=6.6, height=3.8,
        title="Hyperedges give the surface code real short loops",
        subtitle="nonzero loop corrections per weight class "
                 "(repetition d=3 R=3 vs surface d=3 R=3, zero syndrome)",
    )
    weights = sorted(set(rep_counts) | set(surf_counts))
    x = np.arange(len(weights))
    w = 0.32
    for off, counts, color, name in (
        (-w / 2, rep_counts, BLUE, "repetition code"),
        (+w / 2, surf_counts, ORANGE, "surface code"),
    ):
        vals = [counts.get(wt, (0, 0))[0] for wt in weights]
        tots = [counts.get(wt, (0, 0))[1] for wt in weights]
        bars = ax.bar(x + off, vals, width=w - 0.04, color=color, label=name)
        for xi, v, t in zip(x + off, vals, tots):
            ax.annotate(f"{v}/{t}", (xi, v), xytext=(0, 4),
                        textcoords="offset points", fontsize=8.5,
                        color=INK2, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"|l| = {wt}" for wt in weights], fontsize=9)
    ax.set_ylabel("nonzero loops (labels: nonzero / total)", fontsize=10,
                  color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ax.margins(y=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig


def plot_dem_spacetime(demtn, dem, corr_ids, loops_through):
    """Space-time picture of the DEM graph with the correlated events and the
    loops they close.

    demtn: DemTN; dem: the stim DEM (for detector coordinates);
    corr_ids: event indices of the injected correlated events;
    loops_through: iterable of loops (frozensets of TN edge index names) that
    pass through those events.
    """
    coords = dem.get_detector_coordinates()
    det_xy = {d: (c[0], c[1] if len(c) > 1 else 0.0) for d, c in coords.items()}

    def event_xy(i):
        pts = [det_xy[d] for d in demtn.event_dets[i] if d in det_xy]
        if not pts:
            return None
        x = np.mean([p[0] for p in pts])
        y = np.mean([p[1] for p in pts])
        if len(pts) == 1:  # nudge single-detector events off their detector
            x -= 0.55
            y -= 0.22
        return (x, y)

    hot_pairs = set()
    for F in loops_through:
        for ix in F:
            if ix.startswith("e") and "_d" in ix:
                head, rest = ix.split("_d", 1)
                hot_pairs.add((int(head[1:]), int(rest.split("_")[0])))

    fig, ax = _new_fig(
        width=7.8, height=4.8,
        title="Correlated events close new loops in the DEM graph",
        subtitle="gray = base DEM,  orange ◆ = injected correlated events,  "
                 "blue = loops they close",
    )
    ax.yaxis.grid(False)

    for i in range(demtn.num_events):
        exy = event_xy(i)
        if exy is None:
            continue
        hot_event = i in corr_ids
        for d in demtn.event_dets[i]:
            seg_hot = (i, d) in hot_pairs
            color = BLUE if seg_hot else (ORANGE if hot_event else GRID)
            lw = 2.2 if (seg_hot or hot_event) else 1
            z = 3 if (seg_hot or hot_event) else 1
            ax.plot([exy[0], det_xy[d][0]], [exy[1], det_xy[d][1]],
                    color=color, linewidth=lw, zorder=z)
        if hot_event:
            ax.plot(*exy, marker="D", color=ORANGE, markersize=11,
                    markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=8)
        elif any((i, d) in hot_pairs for d in demtn.event_dets[i]):
            ax.plot(*exy, marker="o", color=BLUE, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
        else:
            ax.plot(*exy, marker="o", color=GRAY, markersize=4, zorder=2)

    for d, (x, y) in det_xy.items():
        ax.plot(x, y, marker="s", color=INK2, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=6)

    ax.set_xlabel("space (detector coordinate)", fontsize=10, color=INK2)
    ax.set_ylabel("time (round)", fontsize=10, color=INK2)
    handles = [
        plt.Line2D([], [], marker="s", color=INK2, linestyle="",
                   markersize=7, label="detector"),
        plt.Line2D([], [], marker="o", color=GRAY, linestyle="",
                   markersize=5, label="error event"),
        plt.Line2D([], [], marker="D", color=ORANGE, linestyle="",
                   markersize=8, label="correlated event"),
        plt.Line2D([], [], color=BLUE, linewidth=2.2, label="loop it closes"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK2,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig
