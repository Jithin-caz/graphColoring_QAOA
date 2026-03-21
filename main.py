"""
Efficient Hybrid Variational Quantum Algorithm for Solving Graph Coloring Problem
==================================================================================
Faithful implementation of:
  Liu et al., "Efficient hybrid variational quantum algorithm for solving graph
  coloring problem", arXiv:2504.21335v1 [quant-ph] (2025)

Pipeline (Fig. 1 of the paper):
  (a) Louvain partitioning  → subgraphs + interaction graph
  (b) QAOA per subgraph     → subgraph k-coloring  (Table I pseudo-code)
  (c) Greedy classical      → interaction-graph coloring (fixed backbone)
      Feedback correction   → re-run subgraph QAOA with fixed colors
  (d) Merge + conflict resolution → final global coloring

Key paper equations implemented:
  Eq.(1)  – qubit/vertex limit per subgraph:  V_sub ≤ ⌊Q / ⌈log2(k)⌉⌋
  Eq.(2)  – per-edge conflict function
  Eq.(3)  – cross-subgraph vertex conflict
  Eq.(4)  – C_total = Σ_edge + λΣ_vertex,  λ = |E|/|V|
  Eq.(5)  – isomorphic subgraph color reuse
  Eq.(6)  – color merging (remapping / combination)
  Eq.(7)  – dynamic step-size conflict adjustment
  Eq.(8)  – validity: ∀(u,v)∈E c_u≠c_v  AND  fixed colors respected
  Eq.(9)  – conflict rate ε = |E_conflict|/|E|
  Eq.(10) – H_edge = Σ_{i=0}^{m-1} (I - Z_{u,i}⊗Z_{v,i})
  Eq.(11) – H_fix  = Σ_{i=0}^{m-1} (|2b_i-1|·Z_{u,i}) + m
  Eq.(12) – H_C = λ_edge·Σ H_edge + λ_fix·Σ H_fix
  Eq.(13) – ℓ(θ) = ⟨ψ(θ)|H_C|ψ(θ)⟩
  Table I – full QAOA solve_k_coloring pseudo-code

Requirements (Python 3.13 compatible):
    pip install networkx python-louvain qiskit qiskit-aer scipy numpy
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import community as community_louvain   # python-louvain
import networkx as nx
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer.primitives import Sampler

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION II-A  Color coding
# ═══════════════════════════════════════════════════════════════════════════════
def get_brooks_bound(subgraph: nx.Graph) -> int:
    """Brooks' Theorem: k <= Δ + 1"""
    if subgraph.number_of_nodes() == 0: return 2
    delta = max(dict(subgraph.degree()).values())
    # Standard Brooks bound logic
    return delta + 1

def num_qubits_per_node(k: int) -> int:
    """
    Paper Section II-A / Eq.(1):
    m = ⌈log2(k)⌉ qubits per vertex (minimum 1).
    Binary string c_i = (c_{i1},...,c_{im}) encodes k colors.
    """
    return max(1, math.ceil(math.log2(k))) if k > 1 else 1


def max_subgraph_vertices(Q_available: int, k: int) -> int:
    """Paper Eq.(1): V_sub ≤ ⌊Q / ⌈log2(k)⌉⌋"""
    return Q_available // num_qubits_per_node(k)


def decode_color_binary(bits: List[int], k: int) -> int:
    """
    Paper Table I step (e):
    c_v = (Σ_{i=0}^{m-1} 2^i · s_{v,i}) mod k
    bits[0] = LSB (qubit 0 of that node's block).
    """
    val = sum(b * (2 ** i) for i, b in enumerate(bits))
    return val % k


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION II-B  Conflict functions  (Eqs. 2–9)
# ═══════════════════════════════════════════════════════════════════════════════

def count_edge_conflicts(graph: nx.Graph, coloring: Dict[int, int]) -> int:
    """Paper Eq.(2): conflict(u,v)=1 iff (u,v)∈E and c(u)=c(v)."""
    return sum(
        1 for u, v in graph.edges()
        if coloring.get(u) is not None
        and coloring.get(v) is not None
        and coloring[u] == coloring[v]
    )


def count_vertex_conflicts(communities: Dict[int, List[int]],
                           sub_colorings: Dict[int, Dict[int, int]]) -> int:
    """
    Paper Eq.(3): conflict(u)=1 if node u appears in multiple subgraphs
    Si, Sj with c_{Si}(u) ≠ c_{Sj}(u).
    """
    node_colors: Dict[int, Dict[int, int]] = defaultdict(dict)
    for cid, nodes in communities.items():
        col = sub_colorings.get(cid, {})
        for n in nodes:
            if n in col:
                node_colors[n][cid] = col[n]
    return sum(
        1 for cid_col in node_colors.values()
        if len(set(cid_col.values())) > 1
    )


def total_conflict_cost(graph: nx.Graph,
                        communities: Dict[int, List[int]],
                        sub_colorings: Dict[int, Dict[int, int]],
                        coloring: Dict[int, int]) -> float:
    """
    Paper Eq.(4): C_total = Σ_edge conflict(u,v) + λ·Σ_vertex conflict(u)
    λ = |E|/|V|.  C_total=0 ↔ no conflicts remain.
    """
    lam = graph.number_of_edges() / max(graph.number_of_nodes(), 1)
    return (count_edge_conflicts(graph, coloring)
            + lam * count_vertex_conflicts(communities, sub_colorings))


def conflict_rate(graph: nx.Graph, coloring: Dict[int, int]) -> float:
    """Paper Eq.(9): ε = |E_conflict| / |E|."""
    return count_edge_conflicts(graph, coloring) / max(graph.number_of_edges(), 1)


def is_valid(graph: nx.Graph,
             coloring: Dict[int, int],
             fixed: Optional[Dict[int, int]] = None) -> bool:
    """
    Paper Eq.(8): Valid ⟺ (∀(u,v)∈E  c_u≠c_v) AND (∀u∈fixed  c_u=c_u^fixed).
    """
    for u, v in graph.edges():
        if coloring.get(u) == coloring.get(v):
            return False
    if fixed:
        for u, fc in fixed.items():
            if coloring.get(u) != fc:
                return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION II-C  Hamiltonian  (Eqs. 10–13)
# ═══════════════════════════════════════════════════════════════════════════════

class ProblemHamiltonian:
    """
    Stores H_C as weighted Pauli-Z interaction terms for circuit construction.
    Terms:
      single (coeff, q)       → coeff · Z_q
      double (coeff, qi, qj)  → coeff · Z_qi ⊗ Z_qj
    """
    def __init__(self) -> None:
        self.single: List[Tuple[float, int]] = []
        self.double: List[Tuple[float, int, int]] = []
        self.constant: float = 0.0

    def add_single(self, coeff: float, q: int) -> None:
        self.single.append((coeff, q))

    def add_double(self, coeff: float, qi: int, qj: int) -> None:
        self.double.append((coeff, qi, qj))


def build_hamiltonian(subgraph: nx.Graph,
                      k: int,
                      fixed_colors: Dict[int, int],
                      lambda_edge: float,
                      lambda_fix: float,
                      ) -> Tuple[ProblemHamiltonian, int, List[int]]:
    """
    Paper Eqs.(10-12):

    H_edge^(u,v) = Σ_{i=0}^{m-1} (I - Z_{u,i}⊗Z_{v,i})            Eq.(10)
      → contributes: constant +m  and  double term -Z_{u,i}⊗Z_{v,i}
      (ZZ=+1 for same-bit → conflict penalised; ZZ=-1 for diff-bit → ok)

    H_fix^(u) = Σ_{i=0}^{m-1} (|2b_i^(u)-1| · Z_{u,i}) + m        Eq.(11)
      → (2b_i-1)=+1 if b_i=1, -1 if b_i=0
      → minimum energy when qubit state matches target bit encoding

    H_C = λ_edge·Σ_{(u,v)∈E} H_edge + λ_fix·Σ_{u∈V} H_fix          Eq.(12)

    Paper specifies: λ_edge=2, λ_fix = 1000·|S| (S = set of fixed nodes),
    ensuring λ_fix >> λ_edge so fixed colors take precedence.
    """
    nodes = list(subgraph.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    m = num_qubits_per_node(k)
    n_qubits = len(nodes) * m

    H = ProblemHamiltonian()

    # ── H_edge  Eq.(10) ───────────────────────────────────────────────────────
    for (u, v) in subgraph.edges():
        iu, iv = node_idx[u], node_idx[v]
        H.constant += lambda_edge * m          # +m per edge term
        for i in range(m):
            qu = iu * m + i
            qv = iv * m + i
            H.add_double(-lambda_edge, qu, qv)  # -Z_{u,i}⊗Z_{v,i}

    # ── H_fix  Eq.(11) ────────────────────────────────────────────────────────
    for v, target_c in fixed_colors.items():
        if v not in node_idx:
            continue
        iv = node_idx[v]
        # LSB-first binary encoding of target color
        bits_b = [(target_c >> i) & 1 for i in range(m)]
        H.constant += lambda_fix * m            # +m per fixed node
        for i in range(m):
            sign = 2 * bits_b[i] - 1            # +1 if b=1, -1 if b=0
            q = iv * m + i
            H.add_single(lambda_fix * sign, q)  # (2b_i-1)·Z_{u,i}

    return H, n_qubits, nodes


# ═══════════════════════════════════════════════════════════════════════════════
# QAOA ANSATZ  —  Fig. 2  /  Table I steps (a-c)
# ═══════════════════════════════════════════════════════════════════════════════

def build_qaoa_circuit(H: ProblemHamiltonian,
                       n_qubits: int,
                       p: int) -> QuantumCircuit:
    """
    Paper Fig. 2 / Table I step (c):

    |ψ(γ,β)⟩ = Π_{l=1}^p  e^{-iβ_l H_M}  e^{-iγ_l H_C} |ψ_0⟩

    |ψ_0⟩ = |+⟩^⊗N   (Hadamard on all qubits)
    H_M = Σ_{v,i} X_{v,i}   (transverse mixer)

    U_C(γ_l):
      single Z term  →  RZ(2γ·coeff)
      ZZ term        →  CX · RZ(2γ·coeff) · CX

    U_B(β_l):
      RX(2β) on every qubit
    """
    gamma = ParameterVector("γ", p)
    beta  = ParameterVector("β", p)

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))   # |ψ_0⟩ = |+⟩^⊗N

    for l in range(p):
        # U_C(γ_l)
        for coeff, q in H.single:
            qc.rz(2.0 * gamma[l] * coeff, q)
        for coeff, qi, qj in H.double:
            qc.cx(qi, qj)
            qc.rz(2.0 * gamma[l] * coeff, qj)
            qc.cx(qi, qj)
        # U_B(β_l)
        qc.rx(2.0 * beta[l], range(n_qubits))

    qc.measure_all()
    return qc


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION  ℓ(θ) = ⟨ψ(θ)|H_C|ψ(θ)⟩  —  Eq.(13)
# ═══════════════════════════════════════════════════════════════════════════════

def expectation_value(counts: Dict[int, float],
                      H: ProblemHamiltonian,
                      n_qubits: int) -> float:
    """
    Paper Eq.(13): ℓ(θ) = Σ_s P(s)·E(s)

    For bitstring s, Z_i eigenvalue = 1-2·s_i (+1 for |0⟩, -1 for |1⟩).
    """
    total_e = 0.0
    total_p = 0.0
    for state_int, prob in counts.items():
        # Qiskit: bit q is at position q of state_int
        z = [1 - 2 * ((state_int >> q) & 1) for q in range(n_qubits)]
        e = H.constant
        for coeff, q in H.single:
            e += coeff * z[q]
        for coeff, qi, qj in H.double:
            e += coeff * z[qi] * z[qj]
        total_e += prob * e
        total_p += prob
    return total_e / max(total_p, 1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE I — QAOA  solve_k_coloring
# ═══════════════════════════════════════════════════════════════════════════════

def solve_k_coloring_qaoa(subgraph: nx.Graph,
                          k: int,
                          fixed_colors: Dict[int, int],
                          p: int,
                          shots: int,
                          Q_available: int,
                          lambda_edge: float,
                          lambda_fix: float,
                          max_no_improve: int = 3,
                          ) -> Optional[Dict[int, int]]:
    """
    Paper Table I:  QAOA solve_k_coloring

    (a) Qubit allocation: N_total = |V|·m; skip if > Q_available
    (b) Construct H_C (Eq.12)
    (c) Initialise |ψ_0⟩ = |+⟩^⊗N; build p-layer QAOA ansatz
    (d) Gradient updates θ_{t+1} = θ_t - η·∇_θ ℓ(θ);
        early stop if no improvement in 3 steps OR conflict_edges=0
    (e) Measure; c_v = (Σ 2^i·s_{v,i}) mod k
    (f) Verify Eq.(8); return (k*, C) if valid

    Returns coloring {node: color} or None if skipped.
    """
    nodes = list(subgraph.nodes())
    m = num_qubits_per_node(k)
    n_total = len(nodes) * m

    # Step (a)
    if n_total > Q_available:
        return None
    if len(nodes) == 0:
        return {}

    # Step (b)
    H, n_qubits, ordered_nodes = build_hamiltonian(
        subgraph, k, fixed_colors, lambda_edge, lambda_fix
    )

    # Step (c)
    qc = build_qaoa_circuit(H, n_qubits, p)
    sampler = Sampler()

    # Step (d): COBYLA optimisation with early stopping
    n_params = 2 * p
    theta0 = np.random.uniform(0.0, np.pi, n_params)
    best_theta = theta0.copy()
    best_loss = float("inf")
    no_improve_streak = [0]

    def objective(th: np.ndarray) -> float:
        gv = th[:p]
        bv = th[p:]
        pd = {f"γ[{i}]": gv[i] for i in range(p)}
        pd.update({f"β[{i}]": bv[i] for i in range(p)})

        bound = qc.assign_parameters(pd)
        result = sampler.run([bound], shots=shots).result()
        loss = expectation_value(result.quasi_dists[0], H, n_qubits)

        if loss < best_loss - 1e-4:
            best_theta[:] = th
            no_improve_streak[0] = 0
        else:
            no_improve_streak[0] += 1

        # Early stop: no improvement for 3 steps
        if no_improve_streak[0] >= max_no_improve:
            raise StopIteration("early_stop")

        # Early stop: current best has 0 conflicts
        col_tmp = _decode_coloring(result.quasi_dists[0], n_qubits,
                                   ordered_nodes, m, k)
        if count_edge_conflicts(subgraph, col_tmp) == 0:
            best_theta[:] = th
            raise StopIteration("zero_conflicts")

        return loss

    try:
        res = minimize(objective, theta0, method="COBYLA",
                       options={"maxiter": 300, "rhobeg": 0.5})
        best_theta[:] = res.x
    except StopIteration:
        pass

    # Step (e): final sampling with optimal parameters
    gv = best_theta[:p]
    bv = best_theta[p:]
    pd = {f"γ[{i}]": gv[i] for i in range(p)}
    pd.update({f"β[{i}]": bv[i] for i in range(p)})
    bound = qc.assign_parameters(pd)
    final_counts = sampler.run([bound], shots=shots * 4).result().quasi_dists[0]

    # Pick highest-probability bitstring
    coloring = _decode_coloring(final_counts, n_qubits, ordered_nodes, m, k)
    return coloring


def _decode_coloring(counts: Dict[int, float],
                     n_qubits: int,
                     nodes: List[int],
                     m: int,
                     k: int) -> Dict[int, int]:
    """Decode the most-probable bitstring → color assignment."""
    if not counts:
        return {}
    best_state = max(counts, key=lambda s: counts[s])
    bits = [(best_state >> q) & 1 for q in range(n_qubits)]
    return {
        node: decode_color_binary(bits[vi * m: (vi + 1) * m], k)
        for vi, node in enumerate(nodes)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION III  (a)  Louvain partitioning
# ═══════════════════════════════════════════════════════════════════════════════

def louvain_partition(graph: nx.Graph,
                      k: int,
                      Q_available: int) -> Dict[int, List[int]]:
    """
    Paper Section III (a):
    - Louvain algorithm for balanced community detection (modularity)
    - n_s = max(2, ⌈√N⌉) target subgraph count
    - Degree-centrality: high-degree nodes placed interior of subgraph
    - Subgraphs sorted by size descending (subgraph 1 = largest)
    - Recursively split communities exceeding Eq.(1) limit
    """
    N = graph.number_of_nodes()
    v_max = max_subgraph_vertices(Q_available, k)
    n_s = max(2, math.ceil(math.sqrt(N)))  # paper: n_s = max(2, √N)

    if N == 0:
        return {}

    partition: Dict[int, int] = community_louvain.best_partition(graph)

    # Invert partition
    raw: Dict[int, List[int]] = defaultdict(list)
    for node, cid in partition.items():
        raw[cid].append(node)

    # Sort nodes within each community by degree desc (degree centrality)
    deg = dict(graph.degree())
    for cid in raw:
        raw[cid].sort(key=lambda v: deg[v], reverse=True)

    # Sort communities by size desc ("subgraph 1 contains the largest")
    sorted_comms = sorted(raw.values(), key=len, reverse=True)

    # Recursively split oversized communities
    final: Dict[int, List[int]] = {}
    counter = 0
    for nodes in sorted_comms:
        if len(nodes) <= v_max:
            final[counter] = nodes
            counter += 1
        else:
            sub = graph.subgraph(nodes).copy()
            sub_parts = louvain_partition(sub, k, Q_available)
            for sp_nodes in sub_parts.values():
                final[counter] = sp_nodes
                counter += 1

    return final


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION III  (2)  Interaction graph + classical greedy coloring
# ═══════════════════════════════════════════════════════════════════════════════

def build_interaction_graph(graph: nx.Graph,
                             communities: Dict[int, List[int]]
                             ) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Paper Section III step (2):
    - Collapse each community to a supernode
    - Add edge between supernodes if any cross-community edge exists
    - Apply classical greedy algorithm to color the interaction graph
    Returns (interaction_graph, inter_coloring {cid -> color}).
    """
    node_to_cid = {n: cid for cid, nodes in communities.items() for n in nodes}

    ig = nx.Graph()
    ig.add_nodes_from(communities.keys())
    for u, v in graph.edges():
        cu, cv = node_to_cid.get(u), node_to_cid.get(v)
        if cu is not None and cv is not None and cu != cv:
            ig.add_edge(cu, cv)

    # Classical greedy coloring of interaction graph (step 2 in Fig.1)
    inter_coloring: Dict[int, int] = nx.coloring.greedy_color(
        ig, strategy="largest_first"
    )
    return ig, inter_coloring


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION III  (3)  Feedback correction  (with Eq.5 isomorphism reuse)
# ═══════════════════════════════════════════════════════════════════════════════

def feedback_correction(graph: nx.Graph,
                        communities: Dict[int, List[int]],
                        sub_colorings: Dict[int, Dict[int, int]],
                        inter_coloring: Dict[int, int],
                        k: int,
                        p: int,
                        shots: int,
                        Q_available: int,
                        lambda_edge: float,
                        lambda_fix: float,
                        max_iter: int = 5,
                        ) -> Dict[int, Dict[int, int]]:
    """
    Paper Section III step (3) / Fig.1(c):
    Fix interaction-graph coloring and feed back to re-run QAOA on
    conflicting subgraphs.

    Includes:
    - Cache for intermediate subgraph results
    - Isomorphic subgraph detection (Eq.5): reuse coloring via φ mapping
    - Iterates until C_total=0 or max_iter reached
    """
    # Backbone: map each node to its supernode color (mod k)
    backbone: Dict[int, int] = {
        n: inter_coloring[cid] % k
        for cid, nodes in communities.items()
        for n in nodes
    }

    iso_cache: Dict[int, Tuple[nx.Graph, Dict[int, int]]] = {}  # Eq.(5) cache

    for iteration in range(max_iter):
        # Build current global coloring to measure C_total
        global_col = _merge_simple(communities, sub_colorings, backbone)
        c_total = count_edge_conflicts(graph, global_col)
        print(f"      [Feedback iter {iteration}] edge conflicts = {c_total}")

        if c_total == 0:
            break

        for cid, nodes in communities.items():
            subgraph = graph.subgraph(nodes).copy()
            col = sub_colorings.get(cid, {})

            if count_edge_conflicts(subgraph, col) == 0:
                continue

            # Eq.(5): check if isomorphic to a previously solved subgraph
            reused = False
            for ref_cid, (ref_sub, ref_col) in iso_cache.items():
                if (subgraph.number_of_nodes() == ref_sub.number_of_nodes()
                        and subgraph.number_of_edges() == ref_sub.number_of_edges()
                        and nx.is_isomorphic(subgraph, ref_sub)):
                    gm = nx.algorithms.isomorphism.GraphMatcher(subgraph, ref_sub)
                    mapping = next(gm.isomorphisms_iter(), None)
                    if mapping:
                        # c_{Gi}(v) = c_{Gj}(φ^{-1}(v))   Eq.(5)
                        sub_colorings[cid] = {
                            v: ref_col[mapping[v]] % k for v in nodes
                        }
                        reused = True
                        break

            if reused:
                continue

            # Re-run QAOA with backbone as fixed constraints
            fixed = {n: backbone[n] for n in nodes}
            new_col = solve_k_coloring_qaoa(
                subgraph, k, fixed, p, shots, Q_available,
                lambda_edge, lambda_fix
            )
            if new_col is not None:
                sub_colorings[cid] = new_col
                iso_cache[cid] = (subgraph.copy(), new_col)

    return sub_colorings


def _merge_simple(communities: Dict[int, List[int]],
                  sub_colorings: Dict[int, Dict[int, int]],
                  backbone: Dict[int, int]) -> Dict[int, int]:
    """Quick merge using sub_coloring with backbone fallback."""
    col: Dict[int, int] = {}
    for cid, nodes in communities.items():
        sc = sub_colorings.get(cid, {})
        for n in nodes:
            col[n] = sc.get(n, backbone.get(n, 0))
    return col


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION III  (4)  Color merging  (Eqs. 6, 7)
# ═══════════════════════════════════════════════════════════════════════════════

def merge_colorings(communities: Dict[int, List[int]],
                    sub_colorings: Dict[int, Dict[int, int]],
                    inter_coloring: Dict[int, int],
                    k: int) -> Dict[int, int]:
    """
    Paper Section II-B Eq.(6):
    α_i = base color of subgraph G_i in interaction graph
    k_i = max internal color + 1

    Color synthesis:
      k_i > k:  c_merged(v) = (α_i + c_in(v)) mod k          (remapping)
      k_i ≤ k:  c_merged(v) = (α_i·k_i + c_in(v)) mod (k·k_i) (combination)
    """
    global_col: Dict[int, int] = {}
    for cid, nodes in communities.items():
        col = sub_colorings.get(cid, {})
        alpha_i = inter_coloring.get(cid, 0)
        k_i = (max(col.values()) + 1) if col else 1

        for v in nodes:
            c_in = col.get(v, 0)
            if k_i > k:
                global_col[v] = (alpha_i + c_in) % k                  # Eq.(6a)
            else:
                global_col[v] = (alpha_i * k_i + c_in) % (k * k_i)   # Eq.(6b)

    return global_col


def resolve_conflicts_greedy(graph: nx.Graph,
                              coloring: Dict[int, int],
                              communities: Dict[int, List[int]],
                              k: int,
                              max_iter: int = 20) -> Dict[int, int]:
    """
    Paper Section II-B Eq.(7) + Section III last paragraph:
    "local greedy conflict-correction colouring"

    M = k · max_i(k_i)   (color space modulus)
    Δ(u) = 1 + ⌊c(u)/M⌋  (dynamic step size)
    c_merged(u) = (c(u) + Δ(u)) mod M
    """
    col = dict(coloring)

    # Compute M
    k_i_vals = []
    for cid, nodes in communities.items():
        vals = [col[v] for v in nodes if v in col]
        k_i_vals.append(max(vals) + 1 if vals else 1)
    M = k * max(k_i_vals) if k_i_vals else k

    for step in range(max_iter):
        conflicts = [(u, v) for u, v in graph.edges()
                     if col.get(u) == col.get(v)]
        if not conflicts:
            break
        print(f"      [Conflict resolution step {step}] "
              f"remaining conflicts = {len(conflicts)}")

        for u, v in conflicts:
            cu = col.get(u, 0)
            delta = 1 + (cu // M) if M > 0 else 1    # Eq.(7)
            col[u] = (cu + delta) % M                  # Eq.(7)

    return col


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE  (Fig. 1)
# ═══════════════════════════════════════════════════════════════════════════════

def quantum_graph_coloring(
    graph: nx.Graph,
    k_max: int           = 4,     # maximum colors to try (sweeps k=2..k_max)
    Q_available: int     = 20,    # qubit budget (NISQ; paper uses ~21)
    p: int               = 2,     # QAOA circuit depth
    shots: int           = 1024,
    lambda_edge: float   = 2.0,   # paper: λ_edge = 2
    feedback_iter: int   = 5,     # step (3) max feedback iterations
    resolve_iter: int    = 20,    # step (4) max greedy correction steps
    seed: int            = 42,
) -> Tuple[Dict[int, int], int, float]:
    """
    Full hierarchical hybrid variational quantum algorithm (Fig. 1).

    Sweeps k from 2 to k_max (paper: "evaluate each k sequentially,
    starting from 2").

    Returns
    -------
    (global_coloring, colors_used, conflict_rate_ε)
    """
    np.random.seed(seed)
    print(f"\n{'='*65}")
    print(f"Graph: N={graph.number_of_nodes()}, E={graph.number_of_edges()}")
    print(f"Q={Q_available}  p={p}  k_max={k_max}")
    print('='*65)

    best_col: Dict[int, int] = {}
    best_k = k_max
    best_eps = 1.0
    start_k = min(k_max, get_brooks_bound(graph))
    for k in range(start_k, 1, -1):
        m = num_qubits_per_node(k)
        # Paper: λ_fix = 1000·|S|  where |S| = total fixed nodes
        S_size = graph.number_of_nodes()
        lambda_fix = 1000 * S_size

        print(f"\n{'─'*65}")
        print(f"[k={k}]  m={m} qubits/node  λ_edge={lambda_edge}  "
              f"λ_fix={lambda_fix}")

        # ── (a) Louvain partitioning ──────────────────────────────────────────
        communities = louvain_partition(graph, k, Q_available)
        print(f"\n  (a) Louvain → {len(communities)} subgraphs: "
              + ", ".join(f"S{c}({len(n)}n)" for c, n in communities.items()))

        # ── (2) Interaction graph + greedy backbone ───────────────────────────
        ig, inter_coloring = build_interaction_graph(graph, communities)
        print(f"  (2) Interaction graph: {ig.number_of_nodes()} supernodes, "
              f"{ig.number_of_edges()} edges")
        print(f"      Backbone colors: {inter_coloring}")

        node_to_cid = {n: cid for cid, nodes in communities.items()
                       for n in nodes}
        backbone = {n: inter_coloring[node_to_cid[n]] % k
                    for n in graph.nodes()}

        # ── (b) QAOA on each subgraph ─────────────────────────────────────────
        sub_colorings: Dict[int, Dict[int, int]] = {}
        print(f"\n  (b) Subgraph QAOA:")

        for cid in sorted(communities.keys()):
            nodes = communities[cid]
            subgraph = graph.subgraph(nodes).copy()
            n_q = len(nodes) * m
            fixed = {n: backbone[n] for n in nodes}

            print(f"      Subgraph {cid}: {len(nodes)} nodes, "
                  f"{n_q} qubits", end=" ... ", flush=True)

            col = solve_k_coloring_qaoa(
                subgraph, k, fixed, p, shots, Q_available,
                lambda_edge, lambda_fix
            )
            if col is None:
                print("SKIPPED (exceeds qubit limit)")
                col = fixed  # fallback to backbone
            else:
                lc = count_edge_conflicts(subgraph, col)
                print(f"local conflicts={lc}")
            sub_colorings[cid] = col

        # ── (c) Feedback correction ───────────────────────────────────────────
        print(f"\n  (c) Feedback correction:")
        sub_colorings = feedback_correction(
            graph, communities, sub_colorings, inter_coloring,
            k, p, shots, Q_available, lambda_edge, lambda_fix,
            max_iter=feedback_iter
        )

        # ── (d) Merge + resolve conflicts ─────────────────────────────────────
        print(f"\n  (d) Merging colorings (Eq.6):")
        global_col = merge_colorings(
            communities, sub_colorings, inter_coloring, k
        )
        ec = count_edge_conflicts(graph, global_col)
        print(f"      Edge conflicts after merge: {ec}")

        if ec > 0:
            print(f"      Greedy conflict resolution (Eq.7):")
            global_col = resolve_conflicts_greedy(
                graph, global_col, communities, k, max_iter=resolve_iter
            )
            ec = count_edge_conflicts(graph, global_col)

        eps = conflict_rate(graph, global_col)
        k_used = len(set(global_col.values()))
        valid = is_valid(graph, global_col)

        print(f"\n  Result: colors_used={k_used}  conflicts={ec}  "
              f"ε={eps:.4f}  valid={valid}")

        if eps < best_eps:
            best_col, best_k, best_eps = global_col, k_used, eps

        if valid:
            print(f"\n{'='*65}")
            print(f"✓  Valid {k}-coloring!  colors_used={k_used}  ε={best_eps:.4f}")
            print(f"   Coloring: {global_col}")
            print('='*65)
            return global_col, k_used, eps

    # Best-effort result if no valid coloring found
    print(f"\n{'='*65}")
    print(f"[!] No fully valid coloring within k_max={k_max}.")
    print(f"    Best result: colors_used={best_k}  ε={best_eps:.4f}")
    print('='*65)
    return best_col, best_k, best_eps


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    results = []

    # ── Example 1: 10-node, 12-edge (matches paper dataset) ──────────────────
    print("\n>>> Example 1: 10-node, 12-edge random graph (paper dataset style)")
    G1 = nx.gnm_random_graph(10, 12, seed=0)
    col1, k1, eps1 = quantum_graph_coloring(
        G1, k_max=4, Q_available=20, p=1, shots=512, seed=0
    )
    results.append(("10n-12e", G1, col1, k1, eps1))

    # ── Example 2: 10-node, 16-edge ───────────────────────────────────────────
    print("\n>>> Example 2: 10-node, 16-edge random graph")
    G2 = nx.gnm_random_graph(10, 16, seed=0)
    col2, k2, eps2 = quantum_graph_coloring(
        G2, k_max=4, Q_available=20, p=2, shots=512, seed=0
    )
    results.append(("10n-16e", G2, col2, k2, eps2))

    # ── Example 3: 20-node, 20-edge (paper dataset) ───────────────────────────
    print("\n>>> Example 3: 20-node, 20-edge random graph")
    G3 = nx.gnm_random_graph(20, 20, seed=0)
    col3, k3, eps3 = quantum_graph_coloring(
        G3, k_max=5, Q_available=21, p=2, shots=512, seed=0
    )
    results.append(("20n-20e", G3, col3, k3, eps3))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Graph':<12} {'N':>4} {'E':>4} {'k_used':>6} {'ε':>7}  {'Valid'}")
    print("="*65)
    for name, G, col, k, eps in results:
        v = is_valid(G, col)
        print(f"{name:<12} {G.number_of_nodes():>4} {G.number_of_edges():>4} "
              f"{k:>6} {eps:>7.4f}  {v}")
    print("="*65)