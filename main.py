"""
Hybrid Variational Quantum Graph Coloring  —  Liu et al., arXiv:2504.21335 (2025)
pip install networkx python-louvain qiskit qiskit-aer scipy numpy matplotlib
"""
from __future__ import annotations
import math, warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import community as community_louvain
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no Tcl/Tk / display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer.primitives import Sampler

warnings.filterwarnings("ignore")

# ── helpers ───────────────────────────────────────────────────────────────────
nqpn   = lambda k: max(1, math.ceil(math.log2(k))) if k > 1 else 1  # Eq.(1) m
vmax   = lambda Q, k: Q // nqpn(k)                                   # Eq.(1) V_sub
dec    = lambda bits, k: sum(b << i for i, b in enumerate(bits)) % k # Table I (e)
brooks = lambda G: max(dict(G.degree()).values()) + 1 if G.number_of_nodes() else 2

def edge_conflicts(G: nx.Graph, col: Dict) -> int:          # Eq.(2)
    return sum(1 for u, v in G.edges()
               if col.get(u) is not None and col.get(v) is not None
               and col[u] == col[v])

def conflict_rate(G: nx.Graph, col: Dict) -> float:         # Eq.(9)
    return edge_conflicts(G, col) / max(G.number_of_edges(), 1)

def is_valid(G: nx.Graph, col: Dict, fixed: Dict = None) -> bool:  # Eq.(8)
    if any(col.get(u) is not None and col.get(u) == col.get(v)
           for u, v in G.edges()):
        return False
    return not fixed or all(col.get(u) == fc for u, fc in fixed.items())

# ── Hamiltonian  Eqs.(10–12) ──────────────────────────────────────────────────
class H:
    def __init__(self): self.s, self.d, self.c = [], [], 0.0

def build_H(G: nx.Graph, k: int, fixed: Dict, le: float, lf: float):
    nodes = list(G.nodes()); idx = {v: i for i, v in enumerate(nodes)}
    m = nqpn(k); h = H()
    for u, v in G.edges():
        h.c += le * m
        for i in range(m):
            h.d.append((-le, idx[u]*m+i, idx[v]*m+i))         # Eq.(10)
    for v, tc in fixed.items():
        if v not in idx: continue
        h.c += lf * m
        for i in range(m):
            h.s.append((lf * (2*((tc >> i) & 1) - 1), idx[v]*m+i))  # Eq.(11)
    return h, len(nodes)*m, nodes

def expectation(counts: Dict, h: H, nq: int) -> float:             # Eq.(13)
    te = tp = 0.0
    for s, p in counts.items():
        z = [1 - 2*((s >> q) & 1) for q in range(nq)]
        e = h.c + sum(c*z[q] for c, q in h.s) + sum(c*z[i]*z[j] for c, i, j in h.d)
        te += p*e; tp += p
    return te / max(tp, 1e-12)

# ── QAOA circuit  Fig.2 ───────────────────────────────────────────────────────
def qaoa_circuit(h: H, nq: int, p: int) -> QuantumCircuit:
    γ, β = ParameterVector("γ", p), ParameterVector("β", p)
    qc = QuantumCircuit(nq); qc.h(range(nq))
    for l in range(p):
        for c, q in h.s:
            qc.rz(2*γ[l]*c, q)
        for c, i, j in h.d:
            qc.cx(i, j); qc.rz(2*γ[l]*c, j); qc.cx(i, j)
        qc.rx(2*β[l], range(nq))
    qc.measure_all()
    return qc

def decode_col(counts, nq, nodes, m, k):
    if not counts: return {}
    s = max(counts, key=counts.get)
    bits = [(s >> q) & 1 for q in range(nq)]
    return {v: dec(bits[i*m:(i+1)*m], k) for i, v in enumerate(nodes)}

# ── Table I: solve_k_coloring  (multi-restart for better solutions) ───────────
def solve_qaoa(G: nx.Graph, k: int, fixed: Dict, p: int, shots: int,
               Q: int, le: float, lf: float, restarts: int = 3) -> Optional[Dict]:
    """
    Paper Table I with multi-restart (paper Sec.IV-A: algorithm re-run with
    different initial parameters to improve solution quality).
    """
    nodes = list(G.nodes()); m = nqpn(k)
    if not nodes or len(nodes)*m > Q: return None
    h, nq, ordered = build_H(G, k, fixed, le, lf)
    qc = qaoa_circuit(h, nq, p)
    sampler = Sampler()

    best_col_global = None
    best_ec_global = float("inf")

    for restart in range(restarts):
        θ = np.random.uniform(0, np.pi, 2*p)
        best = θ.copy(); bl = float("inf"); streak = [0]

        def obj(th):
            pd = {f"γ[{i}]": th[i] for i in range(p)}
            pd |= {f"β[{i}]": th[p+i] for i in range(p)}
            res = sampler.run([qc.assign_parameters(pd)], shots=shots).result()
            loss = expectation(res.quasi_dists[0], h, nq)
            col_tmp = decode_col(res.quasi_dists[0], nq, ordered, m, k)
            if loss < bl - 1e-4:
                best[:] = th; streak[0] = 0
            else:
                streak[0] += 1
            # Paper Table I: early stop if 0 conflicts or no improvement in 3 steps
            if streak[0] >= 3 or edge_conflicts(G, col_tmp) == 0:
                best[:] = th; raise StopIteration
            return loss

        try:
            res = minimize(obj, θ, method="COBYLA",
                           options={"maxiter": 300, "rhobeg": 0.5})
            best[:] = res.x
        except StopIteration:
            pass

        pd = {f"γ[{i}]": best[i] for i in range(p)}
        pd |= {f"β[{i}]": best[p+i] for i in range(p)}
        # Paper: sample more shots at optimal θ* for better statistics
        fc = sampler.run([qc.assign_parameters(pd)], shots=shots*4).result().quasi_dists[0]

        # Paper Table I (e): pick best among ALL sampled bitstrings, not just max-prob
        # This exploits the full measurement distribution for a better solution
        candidate_col = None
        candidate_ec = float("inf")
        for state_int, prob in sorted(fc.items(), key=lambda x: -x[1])[:20]:
            bits = [(state_int >> q) & 1 for q in range(nq)]
            trial = {v: dec(bits[i*m:(i+1)*m], k) for i, v in enumerate(ordered)}
            ec = edge_conflicts(G, trial)
            if ec < candidate_ec:
                candidate_col = trial; candidate_ec = ec
                if ec == 0: break

        if candidate_ec < best_ec_global:
            best_col_global = candidate_col
            best_ec_global = candidate_ec
            if best_ec_global == 0:
                break  # perfect solution found, no need for more restarts

    return best_col_global

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
        if len(nodes) <= vmax(Q, k):
            final[ctr] = nodes; ctr += 1
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
        if cu is not None and cv is not None and cu != cv:
            ig.add_edge(cu, cv)
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
                    if mp:  # Eq.(5)
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
            cu = col.get(u, 0)
            col[u] = (cu + 1 + (cu // M)) % M  # Eq.(7)
    return col

# ── Visualization ─────────────────────────────────────────────────────────────
PALETTE = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
           "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4"]

def draw_partition(G: nx.Graph, comms: Dict, inter: Dict,
                   cross_edges: List[Tuple], title: str):
    """
    Visualise the Louvain partition showing:
    - Each subgraph with its nodes coloured by community
    - Intra-subgraph edges (solid)
    - Cross-subgraph (inter-community) edges (dashed red)
    - Interaction graph summary inset
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    pos = nx.spring_layout(G, seed=42)

    # ── Left: full graph with partition overlay ───────────────────────────────
    ax = axes[0]
    ax.set_title("Graph Partition (Louvain)", fontsize=11)
    ax.axis("off")

    n2c = {n: c for c, ns in comms.items() for n in ns}
    node_colors = [PALETTE[n2c[n] % len(PALETTE)] for n in G.nodes()]

    # Draw intra-community edges (solid)
    intra = [(u, v) for u, v in G.edges() if n2c[u] == n2c[v]]
    nx.draw_networkx_edges(G, pos, edgelist=intra, ax=ax,
                           edge_color="#555555", width=1.8, style="solid")

    # Draw cross-community edges (dashed red) — paper Fig.1 dashed lines
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges, ax=ax,
                           edge_color="#e74c3c", width=1.5, style="dashed",
                           alpha=0.8)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=600, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10,
                            font_color="white", font_weight="bold")

    # Legend: community patches
    patches = [mpatches.Patch(color=PALETTE[c % len(PALETTE)],
                               label=f"Subgraph {c}: nodes {ns}")
               for c, ns in comms.items()]
    patches.append(mpatches.Patch(color="#e74c3c",
                                   label="Cross-subgraph edges"))
    ax.legend(handles=patches, loc="lower left", fontsize=8,
              framealpha=0.9)

    # Subgraph boundary annotations
    for cid, ns in comms.items():
        xs = [pos[n][0] for n in ns]; ys = [pos[n][1] for n in ns]
        cx, cy = np.mean(xs), np.mean(ys)
        ax.annotate(f"S{cid}", (cx, cy), fontsize=9, color="black",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc=PALETTE[cid % len(PALETTE)], alpha=0.3))

    # ── Right: interaction graph ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("Interaction Graph (Supernodes)", fontsize=11)
    ax2.axis("off")

    ig, _ = interaction_graph(G, comms)
    ig_pos = nx.spring_layout(ig, seed=0)
    ig_colors = [PALETTE[c % len(PALETTE)] for c in ig.nodes()]

    nx.draw_networkx_edges(ig, ig_pos, ax=ax2,
                           edge_color="#888888", width=2.0)
    nx.draw_networkx_nodes(ig, ig_pos, ax=ax2, node_color=ig_colors,
                           node_size=900, edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(ig, ig_pos, ax=ax2,
                            labels={c: f"S{c}\n{len(comms[c])}n" for c in ig.nodes()},
                            font_size=9, font_color="white", font_weight="bold")

    ig_ec = {(u, v): inter[u] for u, v in ig.edges()}
    ig_edge_labels = {(u, v): f"c({u})={inter[u]}, c({v})={inter[v]}"
                      for u, v in ig.edges()}
    nx.draw_networkx_edge_labels(ig, ig_pos, edge_labels=ig_edge_labels,
                                  ax=ax2, font_size=7, alpha=0.8)

    # Cross-edge count annotation
    ax2.text(0.02, 0.98,
             f"Cross-subgraph edges: {len(cross_edges)}\n"
             f"Supernode edges: {ig.number_of_edges()}",
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    plt.tight_layout()
    plt.savefig("partition_view.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Plot] Partition view saved → partition_view.png")


def draw_final_coloring(G: nx.Graph, col: Dict, title: str):
    """Draw the final valid k-coloring of the graph."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    pos = nx.spring_layout(G, seed=42)

    colors_used = sorted(set(col.values()))
    color_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(colors_used)}
    node_colors = [color_map[col[n]] for n in G.nodes()]

    # Highlight conflict edges in red
    conflict_edges = [(u, v) for u, v in G.edges() if col.get(u) == col.get(v)]
    ok_edges = [(u, v) for u, v in G.edges() if (u, v) not in conflict_edges]

    nx.draw_networkx_edges(G, pos, edgelist=ok_edges, ax=ax,
                           edge_color="#555555", width=1.8)
    if conflict_edges:
        nx.draw_networkx_edges(G, pos, edgelist=conflict_edges, ax=ax,
                               edge_color="red", width=3, style="dashed")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=700, edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11,
                            font_color="white", font_weight="bold")

    patches = [mpatches.Patch(color=color_map[c], label=f"Color {c}")
               for c in colors_used]
    if conflict_edges:
        patches.append(mpatches.Patch(color="red", label="⚠ Conflict edge"))
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    ec = edge_conflicts(G, col)
    eps = conflict_rate(G, col)
    ax.text(0.02, 0.98,
            f"Colors used: {len(colors_used)}\nConflicts: {ec}\nε = {eps:.4f}\nValid: {is_valid(G, col)}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9))

    plt.tight_layout()
    plt.savefig("final_coloring.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Plot] Final coloring saved → final_coloring.png")

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

        # (a) Louvain partition
        comms = louvain_partition(G, k, Q)
        n2c = {n: c for c, ns in comms.items() for n in ns}
        cross_edges = [(u, v) for u, v in G.edges() if n2c[u] != n2c[v]]
        print(f"  (a) {len(comms)} subgraphs: " +
              ", ".join(f"S{c}({len(n)}n)" for c, n in comms.items()))
        print(f"      Cross-subgraph edges: {len(cross_edges)} → {cross_edges}")

        # (2) Interaction graph
        ig, inter = interaction_graph(G, comms)
        print(f"  (2) {ig.number_of_nodes()} supernodes  "
              f"{ig.number_of_edges()} inter-edges  backbone={inter}")
        print(f"      Supernode edges: {list(ig.edges())}")

        # Visualise partition for this k
        draw_partition(G, comms, inter, cross_edges,
                       f"Louvain Partition  (k={k}, {len(comms)} subgraphs)")

        backbone = {n: inter[n2c[n]] % k for n in G.nodes()}

        # (b) QAOA on each subgraph
        print("  (b) Subgraph QAOA:")
        subs = {}
        for cid in sorted(comms):
            ns = comms[cid]; sg = G.subgraph(ns).copy()
            print(f"      S{cid}: nodes={ns}  edges={list(sg.edges())}  "
                  f"{len(ns)*m}q", end=" ... ", flush=True)
            col = solve_qaoa(sg, k, {n: backbone[n] for n in ns},
                             p, shots, Q, le, lf)
            if col is None:
                print("SKIPPED"); col = {n: backbone[n] for n in ns}
            else:
                print(f"conflicts={edge_conflicts(sg, col)}  col={col}")
            subs[cid] = col

        # (c) Feedback
        print("  (c) Feedback:")
        subs = feedback(G, comms, subs, inter, k, p, shots, Q, le, lf, fb_iter)

        # (d) Merge + resolve
        print("  (d) Merge + resolve:")
        gcol = merge(comms, subs, inter, k)
        ec = edge_conflicts(G, gcol)
        print(f"      post-merge conflicts={ec}")
        if ec > 0:
            gcol = resolve(G, gcol, comms, k, res_iter)
            ec = edge_conflicts(G, gcol)

        eps = conflict_rate(G, gcol); k_used = len(set(gcol.values()))
        valid = is_valid(G, gcol)
        print(f"  → colors={k_used} conflicts={ec} ε={eps:.4f} valid={valid}")

        if eps < best_eps:
            best_col, best_k, best_eps = gcol, k_used, eps
        if valid:
            print(f"\n✓ Valid {k}-coloring: {gcol}")
            draw_final_coloring(G, gcol,
                                f"Final {k}-Coloring  (χ={k_used}, ε={eps:.4f})")
            return gcol, k_used, eps

    # Best-effort final plot
    draw_final_coloring(G, best_col,
                        f"Best Coloring  (colors={best_k}, ε={best_eps:.4f})")
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
    print(f"Edges: {list(G.edges())}")
    print(f"Brooks bound: {brooks(G)}")

    col, k_used, eps = quantum_graph_coloring(
        G, k_max=3, Q=20, p=2, shots=512, seed=42
    )

    print(f"\n{'='*60}")
    print(f"Result  →  colors used : {k_used}")
    print(f"           conflict rate: {eps:.4f}")
    print(f"           valid        : {is_valid(G, col)}")
    print(f"           coloring     : {col}")
    print(f"{'='*60}")