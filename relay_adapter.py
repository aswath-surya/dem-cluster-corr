"""Bridge between Tanner-graph BP solvers (RelayBP / FastBP from the
qlpdc_bp_loops repo) and the DEM tensor network's cluster expansion.

The DEM TN and the Tanner graph are the same factor graph in two costumes:
mechanism Bernoulli+COPY = variable node with prior LLR ln((1-p)/p), clamped
XOR tensor = check node with syndrome sign. This module (i) exports the
(H, p, s) triple a Tanner solver needs, (ii) converts a converged sum-product
fixed point (per-edge LLR messages) into the bond-message dictionary quimb's
D1BP uses, so ClusterExpansion can expand around a Relay-selected basin
instead of whatever basin plain BP falls into.

Conventions: LLR L = ln(P(bit=0)/P(bit=1)); a message vector on a bond is
[sigma(L), sigma(-L)] with sigma the logistic function (any positive scale
works, normalize_message_pairs rescales). FastBP edge order is
np.where(H != 0), edge k = (edge_check[k], edge_bit[k]).
"""

import numpy as np


def _vec(L, clip=60.0):
    L = float(np.clip(L, -clip, clip))
    return np.array([1.0, np.exp(-L)]) / (1.0 + np.exp(-L))


def dem_text_from_H(H, p):
    """Code-capacity DEM: one mechanism per column of H at probability p."""
    lines = []
    H = np.asarray(H)
    for j in range(H.shape[1]):
        dets = " ".join(f"D{i}" for i in np.where(H[:, j])[0])
        lines.append(f"error({p}) {dets}")
    return "\n".join(lines)


def tanner_messages_to_tn(demtn, tn, fbp_result):
    """Convert a FastBP fixed point into a D1BP message dict for `tn`.

    demtn: the DemTN that built `tn` (event_dets gives column supports);
    fbp_result: FastBPResult with llr_b2c_arr (bit->check nu), llr_c2b_arr
    (check->bit), llr_bit (posterior LLRs), and the H/edge tables.

    Requires the TN to be chain-free (detector degree <= chain cutoff),
    which holds for code-capacity qLDPC DEMs where detector degree equals
    the check row weight.
    """
    H = np.asarray(fbp_result.H)
    rows, cols = np.where(H != 0)
    b2c = fbp_result.llr_b2c_arr
    c2b = fbp_result.llr_c2b_arr
    llr_bit = fbp_result.llr_bit
    probs = np.asarray(demtn.probs, dtype=float)
    lam = np.log((1.0 - probs) / probs)

    # per-(bit, check) lookup of the two directed messages
    edge_of = {(int(cols[k]), int(rows[k])): k for k in range(len(rows))}

    def tid_of(tag):
        (t,) = tn.tag_map[tag]
        return t

    messages = {}
    for i in range(demtn.num_events):
        tid_b = tid_of(f"event{i}")
        tid_c = tid_of(f"copy{i}")
        main = f"e{i}_main"
        p = probs[i]
        messages[(main, tid_c)] = np.array([1.0 - p, p])
        # total extrinsic evidence into bit i = posterior minus prior
        messages[(main, tid_b)] = _vec(llr_bit[i] - lam[i])
        for j, d in enumerate(demtn.event_dets[i]):
            bond = f"e{i}_d{d}_{j}"
            tid_x = tid_of(f"det{d}")
            k = edge_of[(i, d)]
            messages[(bond, tid_x)] = _vec(b2c[k])   # variable -> check
            messages[(bond, tid_c)] = _vec(c2b[k])   # check -> variable
    return messages


def relay_polished_fixed_point(H, p_prior, syndrome, relay, fbp,
                               polish_iters=60):
    """Relay-BP basin selection + memory-free sum-product polish.

    Returns (fbp_result_or_None, info). The polish is mandatory: relay legs
    are min-sum with memory, and the cluster expansion is only defined
    around sum-product fixed points.
    """
    e_hat = relay.decode(syndrome)
    nu = relay.last_state["nu"]
    r = fbp.run(syndrome, p_prior, max_iter=polish_iters, init_b2c=nu)
    info = dict(relay_found=relay.last_state["found"],
                legs=relay.last_state["legs"],
                polish_converged=bool(r.converged),
                relay_solution=e_hat)
    return (r if r.converged else None), info
