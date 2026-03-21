"""
Hybrid Variational Quantum Graph Coloring  —  Liu et al., arXiv:2504.21335 (2025)
pip install networkx python-louvain qiskit qiskit-aer scipy numpy
"""
from __future__ import annotations
import math, warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import community as community_louvain
import networkx as nx
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer.primitives import Sampler

warnings.filterwarnings("ignore")

# ── helpers ──────────────────────────────────────────────────────────────────
nqpn  = lambda k: max(1, math.ceil(math.log2(k))) if k > 1 else 1   # Eq.(1) m
vmax  = lambda Q, k: Q // nqpn(k)                                    # Eq.(1) V_sub
dec   = lambda bits, k: sum(b << i for i, b in enumerate(bits)) % k  # Table I (e)
brooks = lambda G: max(dict(G.degree()).values()) + 1 if G.number_of_nodes() else 2

def edge_conflicts(G: nx.Graph, col: Dict) -> int:           # Eq.(2)
    return sum(1 for u, v in G.edges() if col.get(u) == col.get(v) is not None)

def conflict_rate(G: nx.Graph, col: Dict) -> float:          # Eq.(9)
    return edge_conflicts(G, col) / max(G.number_of_edges(), 1)

def is_valid(G: nx.Graph, col: Dict, fixed: Dict = None) -> bool:  # Eq.(8)
    if any(col.get(u) == col.get(v) for u, v in G.edges()): return False
    return not fixed or all(col.get(u) == fc for u, fc in fixed.items())

# ── Hamiltonian  Eqs.(10-12) ─────────────────────────────────────────────────
class H:
    def __init__(self): self.s, self.d, self.c = [], [], 0.0

def build_H(G: nx.Graph, k: int, fixed: Dict, le: float, lf: float):
    nodes = list(G.nodes()); idx = {v: i for i, v in enumerate(nodes)}
    m = nqpn(k); h = H()
    for u, v in G.edges():
        h.c += le * m
        for i in range(m): h.d.append((-le, idx[u]*m+i, idx[v]*m+i))  # Eq.(10)
    for v, tc in fixed.items():
        if v not in idx: continue
        h.c += lf * m
        for i in range(m):
            h.s.append((lf * (2*((tc>>i)&1) - 1), idx[v]*m+i))        # Eq.(11)
    return h, len(nodes)*m, nodes

def expectation(counts: Dict, h: H, nq: int) -> float:                 # Eq.(13)
    te = tp = 0.0
    for s, p in counts.items():
        z = [1 - 2*((s>>q)&1) for q in range(nq)]
        e = h.c + sum(c*z[q] for c,q in h.s) + sum(c*z[i]*z[j] for c,i,j in h.d)
        te += p*e; tp += p
    return te / max(tp, 1e-12)

# ── QAOA circuit  Fig.2 ───────────────────────────────────────────────────────
def qaoa_circuit(h: H, nq: int, p: int) -> QuantumCircuit:
    γ, β = ParameterVector("γ", p), ParameterVector("β", p)
    qc = QuantumCircuit(nq); qc.h(range(nq))
    for l in range(p):
        for c, q in h.s: qc.rz(2*γ[l]*c, q)
        for c, i, j in h.d: qc.cx(i,j); qc.rz(2*γ[l]*c, j); qc.cx(i,j)
        qc.rx(2*β[l], range(nq))
    qc.measure_all(); return qc

def decode_col(counts, nq, nodes, m, k):
    if not counts: return {}
    s = max(counts, key=counts.get)
    bits = [(s>>q)&1 for q in range(nq)]
    return {v: dec(bits[i*m:(i+1)*m], k) for i, v in enumerate(nodes)}

# ── Table I: solve_k_coloring ─────────────────────────────────────────────────
def solve_qaoa(G: nx.Graph, k: int, fixed: Dict, p: int, shots: int,
               Q: int, le: float, lf: float) -> Optional[Dict]:
    nodes = list(G.nodes()); m = nqpn(k)
    if not nodes or len(nodes)*m > Q: return None
    h, nq, ordered = build_H(G, k, fixed, le, lf)
    qc = qaoa_circuit(h, nq, p); sampler = Sampler()
    θ = np.random.uniform(0, np.pi, 2*p); best = θ.copy(); bl = float("inf"); streak = [0]

    def obj(th):
        pd = {f"γ[{i}]": th[i] for i in range(p)}
        pd |= {f"β[{i}]": th[p+i] for i in range(p)}
        res = sampler.run([qc.assign_parameters(pd)], shots=shots).result()
        loss = expectation(res.quasi_dists[0], h, nq)
        if loss < bl - 1e-4: best[:] = th; streak[0] = 0
        else: streak[0] += 1
        if streak[0] >= 3: raise StopIteration
        if edge_conflicts(G, decode_col(res.quasi_dists[0], nq, ordered, m, k)) == 0:
            best[:] = th; raise StopIteration
        return loss

    try: res = minimize(obj, θ, method="COBYLA", options={"maxiter":300,"rhobeg":.5}); best[:]=res.x
    except StopIteration: pass

    pd = {f"γ[{i}]": best[i] for i in range(p)} | {f"β[{i}]": best[p+i] for i in range(p)}
    fc = sampler.run([qc.assign_parameters(pd)], shots=shots*4).result().quasi_dists[0]
    return decode_col(fc, nq, ordered, m, k)

# ── Louvain partition  Sec.III(a) ─────────────────────────────────────────────
def louvain_partition(G: nx.Graph, k: int, Q: int) -> Dict[int, List]:
    if not G.number_of_nodes(): return {}
    part = community_louvain.best_partition(G)
    raw: Dict = defaultdict(list)
    for n, c in part.items(): raw[c].append(n)
    deg = dict(G.degree())
    for c in raw: raw[c].sort(key=lambda v: deg[v], reverse=True)
    final, ctr = {}, 0
    for nodes in sorted(raw.values(), key=len, reverse=True):
        if len(nodes) <= vmax(Q, k): final[ctr] = nodes; ctr += 1
        else:
            for sp in louvain_partition(G.subgraph(nodes).copy(), k, Q).values():
                final[ctr] = sp; ctr += 1
    return final

# ── Interaction graph  Sec.III(2) ────────────────────────────────────────────
def interaction_graph(G: nx.Graph, comms: Dict) -> Tuple[nx.Graph, Dict]:
    n2c = {n: c for c, ns in comms.items() for n in ns}
    ig = nx.Graph(); ig.add_nodes_from(comms)
    for u, v in G.edges():
        cu, cv = n2c.get(u), n2c.get(v)
        if cu is not None and cv is not None and cu != cv: ig.add_edge(cu, cv)
    return ig, nx.coloring.greedy_color(ig, strategy="largest_first")

# ── Feedback correction  Sec.III(3) ──────────────────────────────────────────
def feedback(G: nx.Graph, comms: Dict, subs: Dict, inter: Dict,
             k: int, p: int, shots: int, Q: int, le: float, lf: float,
             max_iter: int = 5) -> Dict:
    backbone = {n: inter[c] % k for c, ns in comms.items() for n in ns}
    iso_cache = {}
    for it in range(max_iter):
        merged = {n: subs.get(c, {}).get(n, backbone.get(n, 0))
                  for c, ns in comms.items() for n in ns}
        ec = edge_conflicts(G, merged)
        print(f"      [Feedback {it}] conflicts={ec}")
        if ec == 0: break
        for cid, nodes in comms.items():
            sg = G.subgraph(nodes).copy()
            col = subs.get(cid, {})
            if edge_conflicts(sg, col) == 0: continue
            reused = False
            for rc, (rs, rcol) in iso_cache.items():
                if (sg.number_of_nodes() == rs.number_of_nodes()
                        and sg.number_of_edges() == rs.number_of_edges()
                        and nx.is_isomorphic(sg, rs)):
                    gm = nx.algorithms.isomorphism.GraphMatcher(sg, rs)
                    mp = next(gm.isomorphisms_iter(), None)
                    if mp:
                        subs[cid] = {v: rcol[mp[v]] % k for v in nodes}
                        reused = True; break
            if reused: continue
            nc = solve_qaoa(sg, k, {n: backbone[n] for n in nodes},
                            p, shots, Q, le, lf)
            if nc: subs[cid] = nc; iso_cache[cid] = (sg.copy(), nc)
    return subs

# ── Merge  Eq.(6) ─────────────────────────────────────────────────────────────
def merge(comms: Dict, subs: Dict, inter: Dict, k: int) -> Dict:
    col = {}
    for cid, nodes in comms.items():
        sc = subs.get(cid, {}); α = inter.get(cid, 0)
        ki = max(sc.values()) + 1 if sc else 1
        for v in nodes:
            c_in = sc.get(v, 0)
            col[v] = (α + c_in) % k if ki > k else (α*ki + c_in) % (k*ki)
    return col

# ── Conflict resolution  Eq.(7) ──────────────────────────────────────────────
def resolve(G: nx.Graph, col: Dict, comms: Dict, k: int, max_iter: int = 20) -> Dict:
    col = dict(col)
    ki_max = max((max(col[v] for v in ns if v in col) + 1 if ns else 1)
                 for ns in comms.values())
    M = k * ki_max
    for step in range(max_iter):
        cf = [(u, v) for u, v in G.edges() if col.get(u) == col.get(v)]
        if not cf: break
        print(f"      [Resolve {step}] conflicts={len(cf)}")
        for u, _ in cf:
            cu = col.get(u, 0); col[u] = (cu + 1 + (cu // M)) % M
    return col

# ── Full pipeline  Fig.1 ──────────────────────────────────────────────────────
def quantum_graph_coloring(G: nx.Graph, k_max=4, Q=20, p=2, shots=1024,
                            le=2.0, fb_iter=5, res_iter=20, seed=42):
    np.random.seed(seed)
    print(f"\n{'='*60}\nGraph N={G.number_of_nodes()} E={G.number_of_edges()} "
          f"Q={Q} p={p}\n{'='*60}")
    best_col, best_k, best_eps = {}, k_max, 1.0

    for k in range(min(k_max, brooks(G)), 1, -1):
        m = nqpn(k); lf = 1000 * G.number_of_nodes()
        print(f"\n[k={k}] m={m} λ_edge={le} λ_fix={lf}")

        comms = louvain_partition(G, k, Q)
        print(f"  (a) {len(comms)} subgraphs: " +
              ", ".join(f"S{c}({len(n)}n)" for c, n in comms.items()))

        ig, inter = interaction_graph(G, comms)
        print(f"  (2) {ig.number_of_nodes()} supernodes, backbone={inter}")

        n2c = {n: c for c, ns in comms.items() for n in ns}
        backbone = {n: inter[n2c[n]] % k for n in G.nodes()}

        print("  (b) Subgraph QAOA:")
        subs = {}
        for cid in sorted(comms):
            ns = comms[cid]; sg = G.subgraph(ns).copy()
            print(f"      S{cid}: {len(ns)}n {len(ns)*m}q", end=" ... ", flush=True)
            col = solve_qaoa(sg, k, {n: backbone[n] for n in ns},
                             p, shots, Q, le, lf)
            if col is None: print("SKIPPED"); col = {n: backbone[n] for n in ns}
            else: print(f"conflicts={edge_conflicts(sg, col)}")
            subs[cid] = col

        print("  (c) Feedback:")
        subs = feedback(G, comms, subs, inter, k, p, shots, Q, le, lf, fb_iter)

        print("  (d) Merge + resolve:")
        gcol = merge(comms, subs, inter, k)
        ec = edge_conflicts(G, gcol)
        print(f"      post-merge conflicts={ec}")
        if ec > 0: gcol = resolve(G, gcol, comms, k, res_iter); ec = edge_conflicts(G, gcol)

        eps = conflict_rate(G, gcol); k_used = len(set(gcol.values()))
        valid = is_valid(G, gcol)
        print(f"  → colors={k_used} conflicts={ec} ε={eps:.4f} valid={valid}")

        if eps < best_eps: best_col, best_k, best_eps = gcol, k_used, eps
        if valid:
            print(f"\n✓ Valid {k}-coloring: {gcol}"); return gcol, k_used, eps

    print(f"\n[!] Best: colors={best_k} ε={best_eps:.4f}")
    return best_col, best_k, best_eps


# ── Manual 10-node graph ──────────────────────────────────────────────────────
def make_manual_10node_graph() -> nx.Graph:
    """
    10 nodes, 17 edges. Chromatic number χ = 3.

    Outer pentagon : 0-1, 1-2, 2-3, 3-4, 4-0
    Inner pentagon : 5-6, 6-7, 7-8, 8-9, 9-5
    Irregular spokes: 0-5, 1-7, 2-9, 3-6, 4-8
    Extra chords   : 0-2, 5-8
    """
    G = nx.Graph()
    G.add_nodes_from(range(10))
    G.add_edges_from([
        (0,1),(1,2),(2,3),(3,4),(4,0),   # outer pentagon
        (5,6),(6,7),(7,8),(8,9),(9,5),   # inner pentagon
        (0,5),(1,7),(2,9),(3,6),(4,8),   # irregular spokes
        (0,2),(5,8),                      # extra chords
    ])
    return G


if __name__ == "__main__":
    G = make_manual_10node_graph()
    print(f"Manual graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    col, k_used, eps = quantum_graph_coloring(
        G, k_max=3, Q=20, p=2, shots=512, seed=42
    )

    print(f"\n{'='*60}")
    print(f"Result  →  colors used : {k_used}")
    print(f"           conflict rate: {eps:.4f}")
    print(f"           valid        : {is_valid(G, col)}")
    print(f"           coloring     : {col}")
    print(f"{'='*60}")