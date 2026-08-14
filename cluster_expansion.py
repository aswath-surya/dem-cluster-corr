"""Hand-implemented connected-cluster expansion for tensor network contraction.

Implements Midha & Zhang, "Beyond Belief Propagation: Cluster-Corrected Tensor
Network Contraction with Exponential Convergence" (arXiv:2510.02290), following
the tex source (cluster_main.tex / algo_main.tex / bp_main.tex), with the
normalization conventions of Midha, Sommers, Tindall, Abanin (arXiv:2604.03228,
SM Sec. S1). Built as a thin wrapper on quimb: quimb supplies the BP fixed
point (D1BP) and the tensor plumbing; the loop enumeration, loop corrections,
cluster (multiset) enumeration, Ursell function, and free-energy series are
implemented here from the paper's definitions rather than quimb's built-in
Kikuchi-region machinery.

Pipeline (paper section in parentheses):
  1. BP fixed point via quimb D1BP; normalize message pairs so the bond
     overlap I_vw = mu_{v->w} . mu_{w->v} = 1 (S1.1).
  2. Normalize tensors T~_v = T_v / Z^(v) so the BP vacuum contraction is 1
     (Eq. 12 of 2510.02290); log Z_BP = sum_v log Z^(v) is kept separately.
  3. Enumerate generalized loops = connected EDGE-induced subgraphs with
     min degree 2, weight |l| = #edges <= m (Def. II.1 + Algorithm 1). Note
     this is deliberately not quimb's gen_gloops, which enumerates vertex
     sets; the paper's excitations are edge subsets.
  4. Loop correction Z_l (Def. II.2): oblique excitation projectors
     P^perp = 1 - |mu_{w->v}><mu_{v->w}| on loop edges, incoming messages
     closing every other leg of the touched tensors; untouched tensors
     contribute exactly 1 by the normalization.
  5. Clusters = multisets of loops, connected in the interaction graph
     (edges between incompatible-or-identical loops, Defs. III.2-III.4),
     enumerated by DFS growth (Algorithm 2), truncated at cluster weight
     |W| = sum eta_i |l_i| <= m.
  6. Ursell function phi(W) (Eq. 16): brute-force sum of (-1)^{|E(C)|} over
     connected spanning subgraphs C of the interaction graph, / W!.
  7. F~_m = sum_{connected W, |W|<=m} phi(W) Z_W;  Z ~= Z_BP * exp(F~_m).

Also provides the naive truncated loop-series expansion (Eq. 11) for
comparison -- the divergent method the cluster expansion supersedes.
"""

import math
from collections import deque
from itertools import combinations

import numpy as np
import quimb.tensor as qtn
from quimb.tensor.belief_propagation import D1BP


def _connected_spanning_sign_sum(n, edges):
    """sum over subsets C of `edges` that connect all n vertices of
    (-1)^{|C|}  (inner sum of the Ursell function, Eq. 16)."""
    total = 0
    m = len(edges)
    for r in range(n - 1, m + 1):
        for sub in combinations(range(m), r):
            parent = list(range(n))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            comps = n
            for k in sub:
                i, j = edges[k]
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
                    comps -= 1
            if comps == 1:
                total += (-1) ** r
    return total


def dem_supernode_groups(tn):
    """tid -> supernode key for a DemTN-built network: all tensors of one
    detector's (or observable's) parity-chain tree share its role tag
    (det{d} / obs{k}), so grouping by role tag re-merges each chain into a
    single vertex. COPY and Bernoulli tensors are their own supernodes;
    obscap tensors (and anything untagged) stay solo."""
    groups = {}
    for tid, t in tn.tensor_map.items():
        role = None
        for tag in sorted(t.tags):
            if tag != "obscap" and tag.startswith(("det", "obs", "copy", "event")):
                role = tag
                break
        groups[tid] = role if role is not None else f"solo{tid}"
    return groups


class ClusterExpansion:
    """Cluster-corrected BP contraction of a (simple-graph, closed) TN."""

    def __init__(self, tn, max_iterations=1000, tol=5e-9, messages=None,
                 **bp_opts):
        """If `messages` is given (dict (ix, tid) -> length-2 array covering
        both directions of every bond), it is taken as the BP fixed point and
        no message passing is run -- e.g. a fixed point selected by Relay-BP
        on the equivalent Tanner graph and polished with sum-product. The
        cluster expansion then expands around that basin instead of the one
        plain BP happens to fall into."""
        self.tn = tn.copy()
        self.bp = D1BP(self.tn, **bp_opts)
        if messages is None:
            self.bp.run(max_iterations=max_iterations, tol=tol)
        else:
            self.bp.messages.update(messages)
        self.bp.normalize_message_pairs()  # I_vw = 1 on every bond

        # true edges: indices shared by exactly two tensors
        self.edges = {}
        for ix, tids in self.tn.ind_map.items():
            if len(tids) == 2:
                self.edges[ix] = tuple(sorted(tids))

        # local BP factors Z^(v) and normalized tensors (Eq. 12)
        self.sign = 1.0
        self.log_zbp = 0.0
        self.norm_tensors = {}
        for tid, t in self.tn.tensor_map.items():
            zv = self.bp.local_tensor_contract(tid)
            self.sign *= math.copysign(1.0, zv)
            self.log_zbp += math.log(abs(zv))
            self.norm_tensors[tid] = t.copy() / zv

        # oblique excitation projectors P^perp per edge (bp_main.tex Eq. 7):
        # P[a, b] = delta_ab - mu_{w->v}[a] mu_{v->w}[b], a on tv's leg,
        # b on tw's leg, with messages normalized so the overlap is 1
        self.pperp = {}
        for ix, (ta, tb) in self.edges.items():
            m_in_a = self.bp.messages[ix, ta]
            m_in_b = self.bp.messages[ix, tb]
            d = len(m_in_a)
            self.pperp[ix] = np.eye(d) - np.outer(m_in_a, m_in_b)

        self._zl_cache = {}

    # ---- step 3: generalized loop enumeration (edge subsets) ----

    def _core_adj(self):
        """2-core edge set and edge adjacency (edges sharing a vertex)."""
        deg = {}
        vedges = {}
        for ix, (ta, tb) in self.edges.items():
            for t in (ta, tb):
                deg[t] = deg.get(t, 0) + 1
                vedges.setdefault(t, set()).add(ix)
        removed = set()
        queue = deque(t for t, d in deg.items() if d <= 1)
        while queue:
            t = queue.popleft()
            if t in removed:
                continue
            removed.add(t)
            for ix in vedges[t]:
                ta, tb = self.edges[ix]
                other = tb if ta == t else ta
                if other not in removed:
                    vedges[other].discard(ix)
                    deg[other] -= 1
                    if deg[other] <= 1:
                        queue.append(other)
        core_edges = [
            ix for ix, (ta, tb) in self.edges.items()
            if ta not in removed and tb not in removed
        ]
        adj = {}  # edge -> neighboring core edges (sharing a vertex)
        vert_to_edges = {}
        for ix in core_edges:
            for t in self.edges[ix]:
                vert_to_edges.setdefault(t, []).append(ix)
        for ix in core_edges:
            nbrs = set()
            for t in self.edges[ix]:
                nbrs.update(vert_to_edges[t])
            nbrs.discard(ix)
            adj[ix] = nbrs
        return core_edges, adj

    def gen_loops(self, max_weight):
        """All connected edge subsets F, |F| <= max_weight, min degree 2."""
        core_edges, adj = self._core_adj()
        loops = []
        visited = set()
        for start in sorted(core_edges):
            F0 = frozenset([start])
            if F0 not in visited:
                visited.add(F0)
                queue = deque([F0])
            else:
                continue
            while queue:
                F = queue.popleft()
                dcount = {}
                for ix in F:
                    for t in self.edges[ix]:
                        dcount[t] = dcount.get(t, 0) + 1
                n_deg1 = sum(1 for d in dcount.values() if d == 1)
                if n_deg1 == 0 and len(F) > 1:
                    loops.append(F)
                if len(F) >= max_weight:
                    continue
                # prune: each added edge lowers the dangling count by <= 2
                if n_deg1 > 2 * (max_weight - len(F)):
                    continue
                for ix in F:
                    for nix in adj[ix]:
                        nF = F | {nix}
                        if len(nF) <= max_weight and nF not in visited:
                            visited.add(nF)
                            queue.append(nF)
        return loops

    def gen_loops_touching(self, max_weight, tids):
        """Generalized loops containing at least one edge incident to a
        tensor in `tids` (same objects as gen_loops, restricted to those
        touching the given region). For difference objects -- model
        validation LLRs between two DEMs differing only at some mechanisms,
        or pinned-observable ratios -- clusters not touching the modified
        region have (near-)identical corrections in both networks and cancel,
        so only these loops are needed; the restriction also makes larger
        max_weight affordable."""
        tids = set(tids)
        core_edges, adj = self._core_adj()
        seeds = [ix for ix in core_edges
                 if tids.intersection(self.edges[ix])]
        loops = []
        visited = set()
        for start in sorted(seeds):
            F0 = frozenset([start])
            if F0 in visited:
                continue
            visited.add(F0)
            queue = deque([F0])
            while queue:
                F = queue.popleft()
                dcount = {}
                for ix in F:
                    for t in self.edges[ix]:
                        dcount[t] = dcount.get(t, 0) + 1
                n_deg1 = sum(1 for d in dcount.values() if d == 1)
                if n_deg1 == 0 and len(F) > 1:
                    loops.append(F)
                if len(F) >= max_weight:
                    continue
                if n_deg1 > 2 * (max_weight - len(F)):
                    continue
                for ix in F:
                    for nix in adj[ix]:
                        nF = F | {nix}
                        if len(nF) <= max_weight and nF not in visited:
                            visited.add(nF)
                            queue.append(nF)
        return loops

    # ---- step 3b: supernode loop enumeration (chain trees re-merged) ----

    def _supernode_graph(self, groups):
        """Shared machinery of the grouped enumerators: split decomposed-graph
        edges into external/internal(chain), build the bipartite supernode
        adjacency `gadj`, the pair->edge-index map `eix`, and the same-side
        shared-neighbor lists `shared`."""
        ext, internal_adj = {}, {}
        for ix, (ta, tb) in self.edges.items():
            ga, gb = groups[ta], groups[tb]
            if ga == gb:
                internal_adj.setdefault(ga, {}).setdefault(ta, []).append((tb, ix))
                internal_adj[ga].setdefault(tb, []).append((ta, ix))
            else:
                ext[ix] = (ga, gb)
        color, gadj = {}, {}
        for ix, (ga, gb) in ext.items():
            gadj.setdefault(ga, {}).setdefault(gb, []).append(ix)
            gadj.setdefault(gb, {}).setdefault(ga, []).append(ix)
        for g0 in gadj:
            if g0 in color:
                continue
            color[g0] = 0
            dq = deque([g0])
            while dq:
                u = dq.popleft()
                for v in gadj[u]:
                    if v not in color:
                        color[v] = 1 - color[u]
                        dq.append(v)
                    else:
                        assert color[v] != color[u], "supernode graph not bipartite"
        for a in gadj:
            for b in gadj[a]:
                assert len(gadj[a][b]) == 1, "parallel supernode edges unsupported"
        eix = {frozenset((a, b)): gadj[a][b][0] for a in gadj for b in gadj[a]}
        shared = {}
        for x in gadj:
            nb = sorted(gadj[x])
            for u, w in combinations(nb, 2):
                shared.setdefault((u, w), []).append(x)
        return internal_adj, gadj, eix, shared

    def _expand_supernode_cycles(self, cycles, groups, internal_adj, max_weight):
        """Expand each supernode cycle back to its unique decomposed-graph
        realization (chain trees are trees: the internal Steiner connector is
        unique). Returns (expanded_frozenset, supernode_weight) pairs."""
        def steiner(g, terminals):
            if len(terminals) <= 1 or g not in internal_adj:
                return ()
            adjg = internal_adj[g]
            t0, *rest = sorted(terminals)
            parent, pedge, dq = {t0: None}, {}, deque([t0])
            while dq:
                u = dq.popleft()
                for v, ix in adjg.get(u, ()):
                    if v not in parent:
                        parent[v], pedge[v] = u, ix
                        dq.append(v)
            out = set()
            for t in rest:
                u = t
                while u != t0:
                    out.add(pedge[u])
                    u = parent[u]
            return out

        result = []
        for cyc in sorted(cycles, key=sorted):
            if len(cyc) > max_weight:
                continue
            touch = {}
            for ix in cyc:
                for tid in self.edges[ix]:
                    touch.setdefault(groups[tid], set()).add(tid)
            F = set(cyc)
            for g, tids in touch.items():
                F |= set(steiner(g, tids))
            result.append((frozenset(F), len(cyc)))
        return result

    def gen_loops_grouped(self, max_weight, groups):
        """Generalized loops enumerated in the SUPERNODE graph: tensors with
        equal groups[tid] (e.g. all members of one detector's parity-chain
        tree, see dem_supernode_groups) are merged into a single vertex, and
        loop weight counts only inter-supernode edges, i.e. physical
        mechanism-detector adjacencies, undoing the metric distortion the
        chain decomposition introduces.

        Enumeration is closed-form rather than BFS: the supernode graph is
        bipartite (COPY side vs parity side), where every connected min-deg-2
        edge subset of weight <= 6 is a 4-cycle, a 6-cycle, or a theta-6
        (two same-side vertices joined by three 2-paths); the first object
        outside these families is the weight-7 chorded 6-cycle. Hence the
        max_weight <= 6 assertion; BFS on the hub-dominated supernode graph
        is intractable."""
        assert max_weight <= 6, "closed-form enumeration is complete only to weight 6"
        internal_adj, gadj, eix, shared = self._supernode_graph(groups)
        fs = frozenset
        cycles = set()
        for (u, w), com in shared.items():
            for x, y in combinations(com, 2):
                cycles.add(fs((eix[fs((u, x))], eix[fs((x, w))],
                               eix[fs((w, y))], eix[fs((y, u))])))
            if len(com) >= 3 and max_weight >= 6:
                for x, y, z in combinations(com, 3):
                    cycles.add(fs(eix[fs((a, b))]
                                  for a in (u, w) for b in (x, y, z)))
        if max_weight >= 6:
            sadj = {}
            for (u, w) in shared:
                sadj.setdefault(u, set()).add(w)
                sadj.setdefault(w, set()).add(u)
            for u in sorted(sadj):
                for w in sorted(x for x in sadj[u] if x > u):
                    for v in sorted(x for x in (sadj[u] & sadj[w]) if x > w):
                        for a in shared[(u, w)]:
                            for b in shared[(w, v)]:
                                if b == a:
                                    continue
                                for c in shared[(u, v)]:
                                    if c == a or c == b:
                                        continue
                                    cycles.add(fs((
                                        eix[fs((u, a))], eix[fs((a, w))],
                                        eix[fs((w, b))], eix[fs((b, v))],
                                        eix[fs((v, c))], eix[fs((c, u))])))
        return self._expand_supernode_cycles(cycles, groups, internal_adj,
                                             max_weight)

    def gen_loops_grouped_touching(self, max_weight, groups, touch):
        """Supernode loops restricted to those through at least one group in
        `touch` (e.g. the fired detectors of the current syndrome): the
        syndrome-localized, chain-aware loop selection for circuit-level
        networks. 4-cycles and thetas are filtered from the cheap pair
        enumeration; 6-cycles are walked outward from the touch groups
        directly, so the global (intractably large) triangle census is never
        built."""
        assert max_weight <= 6, "closed-form enumeration is complete only to weight 6"
        touch = set(touch)
        internal_adj, gadj, eix, shared = self._supernode_graph(groups)
        fs = frozenset
        cycles = set()
        for (u, w), com in shared.items():
            for x, y in combinations(com, 2):
                if {u, w, x, y} & touch:
                    cycles.add(fs((eix[fs((u, x))], eix[fs((x, w))],
                                   eix[fs((w, y))], eix[fs((y, u))])))
            if len(com) >= 3 and max_weight >= 6:
                for x, y, z in combinations(com, 3):
                    if {u, w, x, y, z} & touch:
                        cycles.add(fs(eix[fs((a, b))]
                                      for a in (u, w) for b in (x, y, z)))
        if max_weight >= 6:
            for d1 in sorted(touch):
                if d1 not in gadj:
                    continue
                nb1 = sorted(gadj[d1])
                for i1, c1 in enumerate(nb1):
                    for c2 in nb1[i1 + 1:]:
                        for d2 in gadj[c2]:
                            if d2 == d1:
                                continue
                            for c3 in gadj[d2]:
                                if c3 == c1 or c3 == c2:
                                    continue
                                for d3 in gadj[c3]:
                                    if d3 == d1 or d3 == d2:
                                        continue
                                    if c1 in gadj.get(d3, ()):
                                        cycles.add(fs((
                                            eix[fs((c1, d1))], eix[fs((d1, c2))],
                                            eix[fs((c2, d2))], eix[fs((d2, c3))],
                                            eix[fs((c3, d3))], eix[fs((d3, c1))])))
        return self._expand_supernode_cycles(cycles, groups, internal_adj,
                                             max_weight)

    def contract_grouped(self, max_weight, groups=None, loops_w=None,
                         return_info=False):
        """contract() with supernode loop enumeration: weight counts physical
        adjacencies, corrections are evaluated on the chain-expanded loops.
        Pass `loops_w` (from gen_loops_grouped) to reuse an enumeration
        across syndromes; loops depend only on the graph."""
        if loops_w is None:
            loops_w = self.gen_loops_grouped(max_weight, groups)
        loops_w = [(F, w) for F, w in loops_w if w <= max_weight]
        loops = [F for F, _ in loops_w]
        weights = [w for _, w in loops_w]
        zl = [self.loop_correction(F) for F in loops]
        clusters, vsets = self.gen_clusters(loops, max_weight, weights=weights)
        F_m = 0.0
        for cl in clusters:
            phi = self.ursell(cl, vsets)
            if phi == 0.0:
                continue
            zW = 1.0
            for i in cl:
                zW *= zl[i]
            F_m += phi * zW
        val = self.sign * math.exp(self.log_zbp + F_m) \
            if self.log_zbp + F_m < 700 else math.inf
        if return_info:
            return val, {
                "n_loops": len(loops),
                "n_clusters": len(clusters),
                "F_m": F_m,
                "loop_weights": sorted(weights),
                "loop_corrections": dict(zip(map(tuple, loops), zl)),
            }
        return val

    # ---- step 4: loop corrections ----

    def loop_correction(self, F):
        F = frozenset(F)
        if F in self._zl_cache:
            return self._zl_cache[F]
        vs = set()
        for ix in F:
            vs.update(self.edges[ix])
        # local copies of the normalized tensors, keyed by tid so the two
        # ends of each loop edge can be renamed unambiguously
        local = {tid: self.norm_tensors[tid].copy() for tid in vs}
        for tid, t in local.items():
            for ix in tuple(t.inds):
                if ix in F or ix not in self.edges:
                    continue
                # vacuum insert: close the leg with the incoming BP message
                t.vector_reduce_(ix, self.bp.messages[ix, tid])
        tensors = list(local.values())
        for ix in F:
            ta, tb = self.edges[ix]
            new_ix = qtn.rand_uuid()
            local[tb].reindex_({ix: new_ix})
            # P[a, b]: a contracts ta's leg, b contracts tb's leg
            tensors.append(qtn.Tensor(self.pperp[ix], inds=(ix, new_ix)))
        ltn = qtn.TensorNetwork(tensors)
        val = ltn.contract(output_inds=(), optimize="greedy")
        self._zl_cache[F] = val
        return val

    # ---- steps 5-7: clusters, Ursell, series ----

    @staticmethod
    def _incompatible(vs1, vs2):
        return not vs1.isdisjoint(vs2)

    def gen_clusters(self, loops, max_weight, weights=None):
        """Connected clusters (multisets of loop indices) with weight <= m.
        `weights` overrides the default weight |F| = #edges, e.g. with
        supernode weights from gen_loops_grouped."""
        n = len(loops)
        w = [len(F) for F in loops] if weights is None else list(weights)
        vsets = []
        for F in loops:
            vs = set()
            for ix in F:
                vs.update(self.edges[ix])
            vsets.append(frozenset(vs))
        # loop interaction adjacency (incompatible or identical). Pairs whose
        # combined weight exceeds the budget can never appear in one cluster,
        # so skipping them changes nothing downstream (self-pairs for
        # multiset doubling stay in nbrs[i] regardless).
        nbrs = [set([i]) for i in range(n)]
        if n and 2 * min(w) <= max_weight:
            for i in range(n):
                for j in range(i + 1, n):
                    if w[i] + w[j] > max_weight:
                        continue
                    if self._incompatible(vsets[i], vsets[j]):
                        nbrs[i].add(j)
                        nbrs[j].add(i)

        clusters = set()
        stack = [((i,), w[i]) for i in range(n) if w[i] <= max_weight]
        clusters.update(c for c, _ in stack)
        while stack:
            cl, wt = stack.pop()
            grow = set()
            for i in set(cl):
                grow |= nbrs[i]
            for j in grow:
                nwt = wt + w[j]
                if nwt > max_weight:
                    continue
                ncl = tuple(sorted(cl + (j,)))
                if ncl not in clusters:
                    clusters.add(ncl)
                    stack.append((ncl, nwt))
        return sorted(clusters), vsets

    def ursell(self, cluster, vsets):
        """phi(W), Eq. 16 of arXiv:2510.02290."""
        nW = len(cluster)
        if nW == 1:
            return 1.0
        # W! = prod eta_i!
        wfact = 1
        for i in set(cluster):
            wfact *= math.factorial(cluster.count(i))
        # interaction graph on the multiset's nW vertices
        edges = []
        for a in range(nW):
            for b in range(a + 1, nW):
                la, lb = cluster[a], cluster[b]
                if la == lb or self._incompatible(vsets[la], vsets[lb]):
                    edges.append((a, b))
        s = _connected_spanning_sign_sum(nW, edges)
        return s / wfact

    def contract(self, max_weight, return_info=False):
        """Cluster-corrected contraction  Z ~= Z_BP * exp(F~_m)."""
        loops = self.gen_loops(max_weight)
        zl = [self.loop_correction(F) for F in loops]
        clusters, vsets = self.gen_clusters(loops, max_weight)
        F_m = 0.0
        for cl in clusters:
            phi = self.ursell(cl, vsets)
            if phi == 0.0:
                continue
            zW = 1.0
            for i in cl:
                zW *= zl[i]
            F_m += phi * zW
        val = self.sign * math.exp(self.log_zbp + F_m) \
            if self.log_zbp + F_m < 700 else math.inf
        if return_info:
            return val, {
                "n_loops": len(loops),
                "n_clusters": len(clusters),
                "F_m": F_m,
                "loop_weights": sorted(len(F) for F in loops),
                "loop_corrections": dict(zip(map(tuple, loops), zl)),
            }
        return val

    def contract_bp(self):
        return self.sign * math.exp(self.log_zbp)

    def contract_loop_series(self, max_weight):
        """Naive truncated loop series (Eq. 11) -- for comparison only.
        Sums ALL generalized loops of weight <= m, including disconnected
        ones, i.e. products over compatible sets of connected loops."""
        loops = self.gen_loops(max_weight)
        zl = [self.loop_correction(F) for F in loops]
        vsets = []
        for F in loops:
            vs = set()
            for ix in F:
                vs.update(self.edges[ix])
            vsets.append(frozenset(vs))
        n = len(loops)
        w = [len(F) for F in loops]
        # sum over compatible subsets with total weight <= m (DFS)
        total = 0.0
        idx = list(range(n))

        def rec(start, wt, prod, chosen):
            nonlocal total
            for k in range(start, n):
                if wt + w[k] > max_weight:
                    continue
                if any(self._incompatible(vsets[k], vsets[c]) for c in chosen):
                    continue
                total += prod * zl[k]
                rec(k + 1, wt + w[k], prod * zl[k], chosen + [k])

        rec(0, 0, 1.0, [])
        return self.sign * math.exp(self.log_zbp) * (1.0 + total)
