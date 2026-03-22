"""
Quantum Hybrid Variational Graph Coloring — Streamlit UI
Based on: Liu et al., arXiv:2504.21335 (2025)

Run:
    streamlit run streamlit_app.py

Both files must be in the same directory:
    main.py           ← original algorithm (unchanged)
    streamlit_app.py  ← this file (UI only)

pip install streamlit networkx python-louvain qiskit qiskit-aer scipy numpy matplotlib pandas
"""

from __future__ import annotations
import io, time
from collections import defaultdict

import streamlit as st
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Import everything from the original algorithm file ────────────────────────
from main import (
    # helper lambdas / functions
    nqpn, vmax, dec, brooks,
    edge_conflicts, conflict_rate, is_valid,
    # algorithm steps
    build_H, expectation, qaoa_circuit, decode_col,
    solve_qaoa, louvain_partition, interaction_graph,
    feedback, merge, resolve,
    # visualization palette already defined there
    PALETTE,
    # graph generator
    make_manual_10node_graph,
    # full pipeline (used as reference; we wrap it below with logging)
    quantum_graph_coloring,
)

# ── Color name labels (UI only) ───────────────────────────────────────────────
COLOR_NAMES = ["Red", "Blue", "Green", "Orange", "Purple",
               "Teal", "Amber", "Navy", "Pink", "Cyan"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Graph Coloring",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --accent3: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e293b;
}

html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.1;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.25rem;
}
.valid-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.valid-true  { background: #064e3b; color: #34d399; border: 1px solid #10b981; }
.valid-false { background: #450a0a; color: #fca5a5; border: 1px solid #ef4444; }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    padding: 8px 0 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.coloring-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
.color-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    padding: 4px 12px;
    border-radius: 6px;
    font-weight: 700;
}
.algo-step {
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    padding: 8px 12px;
    border-radius: 0 8px 8px 0;
    margin: 4px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text);
}
.log-box {
    background: #060912;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
    max-height: 280px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.6;
}
.stButton>button {
    background: linear-gradient(135deg, var(--accent2), #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
.run-button>button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    width: 100% !important;
    font-size: 1rem !important;
    padding: 0.7rem !important;
}
.title-glow {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 4px;
}
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.stSlider label, .stNumberInput label, .stSelectbox label,
[data-testid="stSidebar"] label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stSlider"] > div > div > div { background: var(--accent2) !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }
div[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI-ONLY HELPERS  (matplotlib figures + graph generators)
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_bytes(fig) -> bytes:
    """Serialize a matplotlib figure to PNG bytes for Streamlit download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0a0e1a")
    buf.seek(0)
    return buf.read()


def make_quick_graph(graph_type: str, n: int, **kwargs) -> nx.Graph:
    """Quick graph generator – all types supported by the sidebar."""
    if graph_type == "Manual 10-node (χ=3)":
        return make_manual_10node_graph()          # ← from main.py
    elif graph_type == "Petersen":
        return nx.petersen_graph()
    elif graph_type == "Complete":
        return nx.complete_graph(n)
    elif graph_type == "Cycle":
        return nx.cycle_graph(n)
    elif graph_type == "Path":
        return nx.path_graph(n)
    elif graph_type == "Wheel":
        return nx.wheel_graph(n)
    elif graph_type == "Random (Erdős–Rényi)":
        return nx.erdos_renyi_graph(n, kwargs.get("prob", 0.3), seed=42)
    elif graph_type == "Barabási–Albert":
        return nx.barabasi_albert_graph(n, kwargs.get("m", 2), seed=42)
    elif graph_type == "Grid":
        side = max(2, int(n ** 0.5))
        G = nx.grid_2d_graph(side, side)
        return nx.relabel_nodes(G, {node: i for i, node in enumerate(G.nodes())})
    elif graph_type == "Star":
        return nx.star_graph(n - 1)
    elif graph_type == "Bipartite":
        n1 = n // 2
        return nx.complete_bipartite_graph(n1, n - n1)
    return nx.path_graph(n)


def draw_graph_preview(G: nx.Graph, title: str = "Graph Preview"):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a"); ax.axis("off")
    ax.set_title(title, fontsize=11, color="#e2e8f0")
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#475569", width=1.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#7c3aed",
                           node_size=500, edgecolors="#0a0e1a", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10,
                            font_color="white", font_weight="bold")
    plt.tight_layout()
    return fig


def draw_partition_ui(G: nx.Graph, comms: dict, inter: dict,
                      cross_edges: list, title: str):
    """Dark-themed partition view (UI version of draw_partition from main.py)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#0a0e1a")
    fig.suptitle(title, fontsize=13, fontweight="bold", color="#e2e8f0")
    pos = nx.spring_layout(G, seed=42)

    # ── Left: full graph with partition overlay ───────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0a0e1a"); ax.axis("off")
    ax.set_title("Graph Partition (Louvain)", fontsize=10, color="#94a3b8")

    n2c = {n: c for c, ns in comms.items() for n in ns}
    node_colors = [PALETTE[n2c[n] % len(PALETTE)] for n in G.nodes()]
    intra = [(u, v) for u, v in G.edges() if n2c[u] == n2c[v]]

    nx.draw_networkx_edges(G, pos, edgelist=intra, ax=ax,
                           edge_color="#475569", width=1.8)
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges, ax=ax,
                           edge_color="#ef4444", width=1.5, style="dashed", alpha=0.85)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=600, edgecolors="#0a0e1a", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10,
                            font_color="white", font_weight="bold")

    patches = [mpatches.Patch(color=PALETTE[c % len(PALETTE)],
                              label=f"S{c} ({len(ns)}n)")
               for c, ns in comms.items()]
    patches.append(mpatches.Patch(color="#ef4444", label="Cross-edges"))
    ax.legend(handles=patches, loc="lower left", fontsize=7,
              framealpha=0.8, facecolor="#111827", labelcolor="#e2e8f0")

    # ── Right: interaction graph ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0a0e1a"); ax2.axis("off")
    ax2.set_title("Interaction Graph (Supernodes)", fontsize=10, color="#94a3b8")

    ig, _ = interaction_graph(G, comms)          # ← from main.py
    ig_pos = nx.spring_layout(ig, seed=0)
    ig_colors = [PALETTE[c % len(PALETTE)] for c in ig.nodes()]

    nx.draw_networkx_edges(ig, ig_pos, ax=ax2, edge_color="#64748b", width=2.0)
    nx.draw_networkx_nodes(ig, ig_pos, ax=ax2, node_color=ig_colors,
                           node_size=900, edgecolors="#0a0e1a", linewidths=2)
    nx.draw_networkx_labels(ig, ig_pos, ax=ax2,
                            labels={c: f"S{c}\n{len(comms[c])}n" for c in ig.nodes()},
                            font_size=8, font_color="white", font_weight="bold")
    ax2.text(0.02, 0.98,
             f"Cross-subgraph edges: {len(cross_edges)}\n"
             f"Supernode edges: {ig.number_of_edges()}",
             transform=ax2.transAxes, fontsize=8, va="top", color="#e2e8f0",
             bbox=dict(boxstyle="round", fc="#1a2235", alpha=0.9))
    plt.tight_layout()
    return fig


def draw_final_coloring_ui(G: nx.Graph, col: dict, title: str):
    """Dark-themed final coloring figure (UI version of draw_final_coloring)."""
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a"); ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", color="#e2e8f0")
    pos = nx.spring_layout(G, seed=42)

    colors_used = sorted(set(col.values()))
    color_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(colors_used)}
    node_colors = [color_map[col[n]] for n in G.nodes()]

    conflict_edges = [(u, v) for u, v in G.edges() if col.get(u) == col.get(v)]
    ok_edges       = [(u, v) for u, v in G.edges() if (u, v) not in conflict_edges]

    nx.draw_networkx_edges(G, pos, edgelist=ok_edges, ax=ax,
                           edge_color="#475569", width=1.8)
    if conflict_edges:
        nx.draw_networkx_edges(G, pos, edgelist=conflict_edges, ax=ax,
                               edge_color="red", width=3, style="dashed")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=700, edgecolors="#0a0e1a", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11,
                            font_color="white", font_weight="bold")

    patches = [mpatches.Patch(color=color_map[c],
                              label=f"Color {c} ({COLOR_NAMES[c % len(COLOR_NAMES)]})")
               for c in colors_used]
    if conflict_edges:
        patches.append(mpatches.Patch(color="red", label="⚠ Conflict edge"))
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              framealpha=0.9, facecolor="#111827", labelcolor="#e2e8f0")

    ec  = edge_conflicts(G, col)          # ← from main.py
    eps = conflict_rate(G, col)           # ← from main.py
    ax.text(0.02, 0.98,
            f"Colors used: {len(colors_used)}\nConflicts: {ec}"
            f"\nε = {eps:.4f}\nValid: {is_valid(G, col)}",
            transform=ax.transAxes, fontsize=9, va="top", color="#e2e8f0",
            bbox=dict(boxstyle="round", fc="#111827", alpha=0.9))
    plt.tight_layout()
    return fig


def draw_result_analysis(all_results: list):
    """Bar-chart analysis across all k values tried."""
    if not all_results:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#0a0e1a")
    ks     = [r["k_target"]       for r in all_results]
    crs    = [r["conflict_rate"]  for r in all_results]
    k_used = [r["k_used"]         for r in all_results]
    confs  = [r["conflicts"]      for r in all_results]

    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#e2e8f0")

    axes[0].bar(ks, crs,
                color=["#10b981" if c == 0 else "#ef4444" for c in confs],
                width=0.5, edgecolor="#0a0e1a")
    axes[0].set_title("Conflict Rate per k")
    axes[0].set_xlabel("k (colors)"); axes[0].set_ylabel("Conflict Rate ε")

    axes[1].bar(ks, k_used, color="#7c3aed", width=0.5, edgecolor="#0a0e1a")
    axes[1].bar(ks, ks, width=0.5, color="#1e293b",
                edgecolor="#0a0e1a", alpha=0.5, label="k target")
    axes[1].set_title("Colors Used vs Target")
    axes[1].set_xlabel("k target"); axes[1].set_ylabel("Colors Used")
    axes[1].legend(facecolor="#111827", labelcolor="#e2e8f0", fontsize=7)

    bar_colors = ["#10b981" if r["valid"] else "#ef4444" for r in all_results]
    axes[2].bar(ks, [1] * len(ks), color=bar_colors, width=0.5, edgecolor="#0a0e1a")
    for k_v, r in zip(ks, all_results):
        axes[2].text(k_v, 0.5, "✓ Valid" if r["valid"] else "✗ Invalid",
                     ha="center", va="center",
                     fontsize=9, color="white", fontweight="bold")
    axes[2].set_title("Validity per k")
    axes[2].set_xlabel("k (colors)"); axes[2].set_yticks([])

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE WRAPPER  — calls main.py's algorithm, captures log + per-k data
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_with_log(G: nx.Graph, k_max: int, Q: int, p: int,
                          shots: int, le: float, fb_iter: int,
                          res_iter: int, seed: int):
    """
    Wraps the algorithm from main.py step-by-step so the UI can capture:
      - a human-readable log string
      - per-k result dicts for the analysis tab
      - partition figures for the partition-view tab
      - the final coloring figure
    """
    np.random.seed(seed)
    log = []
    log.append("=" * 55)
    log.append(f"  Graph  N={G.number_of_nodes()}  E={G.number_of_edges()}  Q={Q}  p={p}")
    log.append("=" * 55)

    best_col, best_k, best_eps = {}, k_max, 1.0
    all_results   = []
    partition_figs = {}
    final_fig      = None

    for k in range(min(k_max, brooks(G)), 1, -1):   # brooks() from main.py
        m  = nqpn(k)                                  # nqpn()   from main.py
        lf = 1000 * G.number_of_nodes()
        log.append(f"\n[k={k}]  m={m}  λ_edge={le:.2f}  λ_fix={lf:.0f}")

        # ── (a) Louvain partition ─────────────────────────────────────────────
        comms = louvain_partition(G, k, Q)            # from main.py
        n2c   = {n: c for c, ns in comms.items() for n in ns}
        cross_edges = [(u, v) for u, v in G.edges() if n2c[u] != n2c[v]]
        log.append(f"  (a) Louvain: {len(comms)} subgraphs, "
                   f"{len(cross_edges)} cross-edges")
        for c, ns in comms.items():
            log.append(f"      S{c}: nodes={ns}")

        # ── (2) Interaction graph ─────────────────────────────────────────────
        ig, inter = interaction_graph(G, comms)       # from main.py
        log.append(f"  (2) Interaction graph: {ig.number_of_nodes()} supernodes, "
                   f"{ig.number_of_edges()} edges")
        log.append(f"      Backbone colors: {inter}")

        # Store partition figure (UI helper, not from main.py)
        partition_figs[k] = draw_partition_ui(
            G, comms, inter, cross_edges,
            f"Louvain Partition  (k={k}, {len(comms)} subgraphs)"
        )

        backbone = {n: inter[n2c[n]] % k for n in G.nodes()}

        # ── (b) QAOA per subgraph ─────────────────────────────────────────────
        log.append("  (b) QAOA subgraph optimization:")
        subs = {}
        for cid in sorted(comms):
            ns = comms[cid]
            sg = G.subgraph(ns).copy()
            col = solve_qaoa(sg, k, {n: backbone[n] for n in ns},   # from main.py
                             p, shots, Q, le, lf)
            if col is None:
                log.append(f"      S{cid}: SKIPPED (size exceeded Q)")
                col = {n: backbone[n] for n in ns}
            else:
                ec_sub = edge_conflicts(sg, col)                      # from main.py
                log.append(f"      S{cid}: nodes={ns}  conflicts={ec_sub}  col={col}")
            subs[cid] = col

        # ── (c) Feedback correction ───────────────────────────────────────────
        log.append(f"  (c) Feedback correction ({fb_iter} iters):")
        subs = feedback(G, comms, subs, inter, k, p,                  # from main.py
                        shots, Q, le, lf, fb_iter)

        # ── (d) Merge + resolve ───────────────────────────────────────────────
        log.append("  (d) Merge & resolve:")
        gcol = merge(comms, subs, inter, k)                           # from main.py
        ec   = edge_conflicts(G, gcol)
        log.append(f"      Post-merge conflicts: {ec}")
        if ec > 0:
            gcol = resolve(G, gcol, comms, k, res_iter)               # from main.py
            ec   = edge_conflicts(G, gcol)
            log.append(f"      Post-resolve conflicts: {ec}")

        eps    = conflict_rate(G, gcol)                               # from main.py
        k_used = len(set(gcol.values()))
        valid  = is_valid(G, gcol)                                    # from main.py
        log.append(f"  → colors={k_used}  conflicts={ec}  ε={eps:.4f}  valid={valid}")

        all_results.append({
            "k_target": k, "k_used": k_used,
            "conflicts": ec, "conflict_rate": eps,
            "valid": valid, "coloring": gcol,
            "comms": comms, "inter": inter,
        })

        if eps < best_eps:
            best_col, best_k, best_eps = gcol, k_used, eps

        if valid:
            log.append(f"\n✓ Valid {k}-coloring found: {gcol}")
            final_fig = draw_final_coloring_ui(
                G, gcol,
                f"Final {k}-Coloring  (χ={k_used}, ε={eps:.4f})"
            )
            return (gcol, k_used, eps,
                    "\n".join(log), all_results, partition_figs, final_fig)

    # ── Best-effort fallback ──────────────────────────────────────────────────
    if not best_col:
        best_col = {n: 0 for n in G.nodes()}
        best_k   = 1
        best_eps = conflict_rate(G, best_col)

    final_fig = draw_final_coloring_ui(
        G, best_col,
        f"Best Coloring  (colors={best_k}, ε={best_eps:.4f})"
    )
    log.append(f"\n[!] Best effort: colors={best_k}  ε={best_eps:.4f}")
    return (best_col, best_k, best_eps,
            "\n".join(log), all_results, partition_figs, final_fig)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "graph":          None,
        "edges":          [],
        "num_vertices":   6,
        "result":         None,
        "log":            "",
        "all_results":    [],
        "partition_figs": {},
        "final_fig":      None,
        "graph_name":     "Custom",
        "elapsed":        0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="title-glow">⚛ QuantumColor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Hybrid Variational Graph Coloring<br>'
        'Liu et al. (2025)</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Quick Graph Generator ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Quick Graph Generator</div>',
                unsafe_allow_html=True)
    GRAPH_TYPES = [
        "Manual 10-node (χ=3)", "Petersen", "Complete", "Cycle",
        "Path", "Wheel", "Star", "Bipartite", "Grid",
        "Random (Erdős–Rényi)", "Barabási–Albert",
    ]
    selected_graph_type = st.selectbox("Graph Type", GRAPH_TYPES)
    gen_n = st.slider("Number of vertices", 4, 20, 8, key="gen_n")

    extra_kwargs = {}
    if selected_graph_type == "Random (Erdős–Rényi)":
        extra_kwargs["prob"] = st.slider("Edge probability", 0.1, 0.9, 0.3, 0.05)
    elif selected_graph_type == "Barabási–Albert":
        extra_kwargs["m"] = st.slider("Edges per new node (m)", 1, 4, 2)

    if st.button("🔄 Generate Graph"):
        G_new = make_quick_graph(selected_graph_type, gen_n, **extra_kwargs)
        st.session_state.graph      = G_new
        st.session_state.edges      = list(G_new.edges())
        st.session_state.num_vertices = G_new.number_of_nodes()
        st.session_state.graph_name = selected_graph_type
        st.session_state.result     = None
        st.session_state.all_results = []
        st.rerun()

    st.markdown("---")

    # ── Algorithm Parameters ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Algorithm Parameters</div>',
                unsafe_allow_html=True)
    k_max    = st.slider("Max colors (k_max)",         2,    8,    4)
    Q        = st.slider("Max qubits (Q)",             4,   40,   20)
    p_layers = st.slider("QAOA layers (p)",            1,    5,    2)
    shots    = st.select_slider("Shots",
                   options=[128, 256, 512, 1024, 2048, 4096], value=512)
    le       = st.slider("λ_edge (Hamiltonian penalty)", 0.5, 5.0, 2.0, 0.5)
    fb_iter  = st.slider("Feedback iterations",        1,   10,    5)
    res_iter = st.slider("Resolve iterations",         5,   50,   20)
    seed     = st.number_input("Random seed",          0, 9999,   42)

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
        'color:#475569;line-height:1.8">'
        "📄 arXiv:2504.21335 (2025)<br>"
        "🔧 Qiskit Aer Sampler<br>"
        "🧮 COBYLA optimizer</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<h1 class="title-glow" style="font-size:1.6rem">'
    "Quantum Hybrid Variational Graph Coloring</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle" style="margin-bottom:1.2rem">'
    "Louvain Partitioning + QAOA + Feedback Correction — arXiv:2504.21335</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "🕸️ Graph Builder",
    "⚛️ Run Algorithm",
    "📊 Result Analysis",
    "🎨 Final Coloring",
    "🗂️ Partition Views",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Graph Builder
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("### Graph Configuration")
        n_vertices = st.number_input(
            "Number of vertices", 2, 30,
            st.session_state.num_vertices,
            key="n_vert_input",
        )
        st.session_state.num_vertices = n_vertices

        st.markdown('<div class="section-header">Edge Assignment</div>',
                    unsafe_allow_html=True)
        st.caption("Add edges one by one or use a quick generator from the sidebar.")

        ec1, ec2, ec3 = st.columns([1, 1, 1])
        with ec1:
            u_node = st.number_input("From vertex", 0, n_vertices - 1, 0, key="eu")
        with ec2:
            v_node = st.number_input("To vertex",   0, n_vertices - 1, 1, key="ev")
        with ec3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add Edge"):
                if u_node != v_node:
                    e = (min(u_node, v_node), max(u_node, v_node))
                    if e not in st.session_state.edges:
                        st.session_state.edges.append(e)
                        st.rerun()

        if st.session_state.edges:
            st.markdown('<div class="section-header">Current Edges</div>',
                        unsafe_allow_html=True)
            edge_cols = st.columns(4)
            for i, e in enumerate(st.session_state.edges):
                with edge_cols[i % 4]:
                    if st.button(f"✕ ({e[0]},{e[1]})", key=f"del_e_{i}"):
                        st.session_state.edges.remove(e)
                        st.rerun()

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("🔗 Build Graph"):
                G_new = nx.Graph()
                G_new.add_nodes_from(range(n_vertices))
                G_new.add_edges_from(st.session_state.edges)
                st.session_state.graph      = G_new
                st.session_state.graph_name = "Custom"
                st.session_state.result     = None
                st.success("Graph built!")
                st.rerun()
        with bc2:
            if st.button("🗑️ Clear"):
                st.session_state.edges = []
                st.session_state.graph = None
                st.rerun()

        # Preset edge sets
        st.markdown('<div class="section-header">Preset Edge Sets</div>',
                    unsafe_allow_html=True)
        preset_cols = st.columns(3)
        with preset_cols[0]:
            if st.button("Triangle"):
                st.session_state.edges = [(0, 1), (1, 2), (0, 2)]
                st.rerun()
        with preset_cols[1]:
            if st.button("K4"):
                st.session_state.edges = [
                    (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
                st.rerun()
        with preset_cols[2]:
            if st.button("Pentagon"):
                st.session_state.edges = [(0,1),(1,2),(2,3),(3,4),(4,0)]
                st.rerun()

    with col_right:
        st.markdown("### Graph Preview")
        if st.session_state.graph is not None:
            G = st.session_state.graph
            fig = draw_graph_preview(
                G,
                f"{st.session_state.graph_name}  "
                f"(N={G.number_of_nodes()}, E={G.number_of_edges()})",
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">'
                    f'{G.number_of_nodes()}</div>'
                    f'<div class="metric-label">Vertices</div></div>',
                    unsafe_allow_html=True)
            with mc2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">'
                    f'{G.number_of_edges()}</div>'
                    f'<div class="metric-label">Edges</div></div>',
                    unsafe_allow_html=True)
            with mc3:
                b = brooks(G)     # from main.py
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{b}</div>'
                    f'<div class="metric-label">Brooks Bound</div></div>',
                    unsafe_allow_html=True)
            with mc4:
                try:
                    chi_n = len(set(
                        nx.coloring.greedy_color(G, strategy="largest_first").values()
                    ))
                except Exception:
                    chi_n = "?"
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{chi_n}</div>'
                    f'<div class="metric-label">Greedy χ</div></div>',
                    unsafe_allow_html=True)

            st.markdown("**Adjacency List:**")
            adj_str = "  ".join(
                f"{n}→[{','.join(str(nb) for nb in G.neighbors(n))}]"
                for n in sorted(G.nodes())
            )
            st.code(adj_str, language=None)
        else:
            st.info("👈 Generate a graph from the sidebar or build one manually.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Run Algorithm
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    if st.session_state.graph is None:
        st.warning("⚠️ Build or generate a graph first (Tab 1 or sidebar).")
    else:
        G = st.session_state.graph
        col_cfg, col_run = st.columns([1, 1.4])

        with col_cfg:
            st.markdown("### Parameters Summary")
            params = {
                "Graph":         f"{st.session_state.graph_name} "
                                 f"(N={G.number_of_nodes()}, E={G.number_of_edges()})",
                "Max colors k_max": k_max,
                "Max qubits Q":  Q,
                "QAOA layers p": p_layers,
                "Shots":         shots,
                "λ_edge":        le,
                "Feedback iters": fb_iter,
                "Resolve iters": res_iter,
                "Random seed":   seed,
            }
            for k, v in params.items():
                st.markdown(
                    f'<div class="algo-step"><b>{k}</b>: {v}</div>',
                    unsafe_allow_html=True)

            qubits_needed = G.number_of_nodes() * nqpn(k_max)   # nqpn from main.py
            if qubits_needed > Q:
                st.warning(
                    f"⚠️ Graph may need ~{qubits_needed} qubits for k={k_max}. "
                    f"Increase Q or reduce k_max.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="run-button">', unsafe_allow_html=True)
            run_btn = st.button("🚀 Run Quantum Coloring", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_run:
            st.markdown("### Execution Log")
            if run_btn:
                with st.spinner("⚛️ Running QAOA optimization…"):
                    t0 = time.time()
                    (col_res, k_used, eps,
                     log_txt, all_results,
                     part_figs, fin_fig) = run_pipeline_with_log(
                        G,
                        k_max=k_max, Q=Q, p=p_layers,
                        shots=shots, le=le,
                        fb_iter=fb_iter, res_iter=res_iter,
                        seed=int(seed),
                    )
                    elapsed = time.time() - t0
                    st.session_state.result         = (col_res, k_used, eps)
                    st.session_state.log            = log_txt
                    st.session_state.all_results    = all_results
                    st.session_state.partition_figs = part_figs
                    st.session_state.final_fig      = fin_fig
                    st.session_state.elapsed        = elapsed

            if st.session_state.log:
                st.markdown(
                    f'<div class="log-box">{st.session_state.log}</div>',
                    unsafe_allow_html=True)

        if st.session_state.result is not None:
            col_res, k_used, eps = st.session_state.result
            valid = is_valid(G, col_res)          # from main.py
            st.markdown("---")
            st.markdown("### 🏆 Best Result")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{k_used}</div>'
                    f'<div class="metric-label">Colors Used</div></div>',
                    unsafe_allow_html=True)
            with m2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">'
                    f'{edge_conflicts(G, col_res)}</div>'      # from main.py
                    f'<div class="metric-label">Conflicts</div></div>',
                    unsafe_allow_html=True)
            with m3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">'
                    f'{eps:.4f}</div>'
                    f'<div class="metric-label">Conflict Rate ε</div></div>',
                    unsafe_allow_html=True)
            with m4:
                badge = "valid-true" if valid else "valid-false"
                label = "✓ VALID" if valid else "✗ INVALID"
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">'
                    f'<span class="valid-badge {badge}">{label}</span></div>'
                    f'<div class="metric-label">Coloring</div></div>',
                    unsafe_allow_html=True)
            with m5:
                elapsed = st.session_state.get("elapsed", 0.0)
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">'
                    f'{elapsed:.1f}s</div>'
                    f'<div class="metric-label">Runtime</div></div>',
                    unsafe_allow_html=True)

            st.markdown("**Best Coloring Assignment:**")
            color_html = '<div class="coloring-row">'
            for node, c in sorted(col_res.items()):
                color_html += (
                    f'<span class="color-chip" '
                    f'style="background:{PALETTE[c % len(PALETTE)]}44;'
                    f'border:1px solid {PALETTE[c % len(PALETTE)]};'
                    f'color:{PALETTE[c % len(PALETTE)]}">'
                    f"v{node}→{COLOR_NAMES[c % len(COLOR_NAMES)]}</span>"
                )
            color_html += "</div>"
            st.markdown(color_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Result Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    if not st.session_state.all_results:
        st.info("Run the algorithm first to see result analysis.")
    else:
        all_results = st.session_state.all_results
        G = st.session_state.graph

        st.markdown("### Result Analysis Across k Values")

        df = pd.DataFrame([{
            "k Target":       r["k_target"],
            "k Used":         r["k_used"],
            "Conflicts":      r["conflicts"],
            "Conflict Rate ε": f"{r['conflict_rate']:.4f}",
            "Valid":          "✓" if r["valid"] else "✗",
            "Subgraphs":      len(r["comms"]),
        } for r in all_results])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        analysis_fig = draw_result_analysis(all_results)
        if analysis_fig:
            st.pyplot(analysis_fig, use_container_width=True)
            plt.close(analysis_fig)

        st.markdown("---")
        st.markdown("### Detailed Coloring Per k")
        for r in all_results:
            with st.expander(
                f"k={r['k_target']}  →  {r['k_used']} colors used  |  "
                f"{'✓ Valid' if r['valid'] else '✗ Invalid'}  |  "
                f"ε={r['conflict_rate']:.4f}"
            ):
                col_html = '<div class="coloring-row">'
                for node, c in sorted(r["coloring"].items()):
                    col_html += (
                        f'<span class="color-chip" '
                        f'style="background:{PALETTE[c % len(PALETTE)]}44;'
                        f'border:1px solid {PALETTE[c % len(PALETTE)]};'
                        f'color:{PALETTE[c % len(PALETTE)]}">'
                        f"v{node}→C{c}</span>"
                    )
                col_html += "</div>"
                st.markdown(col_html, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Subgraph partitions:**")
                    for cid, nodes in r["comms"].items():
                        st.markdown(
                            f'<div class="algo-step">S{cid}: vertices {nodes}</div>',
                            unsafe_allow_html=True)
                with c2:
                    st.markdown("**Interaction backbone colors:**")
                    for sn, ic in r["inter"].items():
                        st.markdown(
                            f'<div class="algo-step">'
                            f"Supernode {sn} → color {ic}</div>",
                            unsafe_allow_html=True)

                col_map = r["coloring"]
                if is_valid(G, col_map):                   # from main.py
                    seq = [col_map[v] for v in sorted(col_map.keys())]
                    st.markdown("**Valid coloring sequence (vertex order):**")
                    st.code(str(seq), language=None)
                else:
                    st.markdown("*No fully valid coloring found for this k.*")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Final Coloring
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    if st.session_state.final_fig is None or st.session_state.result is None:
        st.info("Run the algorithm first to see the final coloring.")
    else:
        G = st.session_state.graph
        col_res, k_used, eps = st.session_state.result
        valid = is_valid(G, col_res)                       # from main.py

        c_left, c_right = st.columns([1.6, 1])
        with c_left:
            st.markdown("### Final Coloring Visualization")
            st.pyplot(st.session_state.final_fig, use_container_width=True)
            img_bytes = _fig_to_bytes(st.session_state.final_fig)
            st.download_button(
                "⬇️ Download Final Coloring PNG", img_bytes,
                file_name="final_coloring.png", mime="image/png")

        with c_right:
            st.markdown("### Coloring Details")
            badge_cls = "valid-true" if valid else "valid-false"
            badge_lbl = "✓ VALID COLORING" if valid else "✗ INVALID COLORING"
            st.markdown(
                f'<div style="text-align:center;margin:12px 0">'
                f'<span class="valid-badge {badge_cls}" '
                f'style="font-size:0.9rem;padding:8px 20px">'
                f"{badge_lbl}</span></div>",
                unsafe_allow_html=True)

            mm1, mm2 = st.columns(2)
            with mm1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">'
                    f'{k_used}</div>'
                    f'<div class="metric-label">Colors Used</div></div>',
                    unsafe_allow_html=True)
            with mm2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">'
                    f'{edge_conflicts(G, col_res)}</div>'       # from main.py
                    f'<div class="metric-label">Conflicts</div></div>',
                    unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Node → Color Mapping:**")
            for node, c in sorted(col_res.items()):
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
                    f'<span style="font-family:JetBrains Mono,monospace;color:#94a3b8">'
                    f"v{node}</span>"
                    f'<span style="flex:1;height:1px;background:#1e293b"></span>'
                    f'<span class="color-chip" '
                    f'style="background:{PALETTE[c % len(PALETTE)]}44;'
                    f'border:1px solid {PALETTE[c % len(PALETTE)]};'
                    f'color:{PALETTE[c % len(PALETTE)]}">'
                    f"Color {c} — {COLOR_NAMES[c % len(COLOR_NAMES)]}</span></div>",
                    unsafe_allow_html=True)

            st.markdown("---")
            conflict_edges = [
                (u, v) for u, v in G.edges()
                if col_res.get(u) == col_res.get(v)
            ]
            if conflict_edges:
                st.markdown("**⚠️ Conflicting Edges:**")
                for u, v in conflict_edges:
                    st.markdown(
                        f'<div class="algo-step" style="border-color:#ef4444">'
                        f"({u}, {v}) — both Color {col_res[u]}</div>",
                        unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="algo-step" style="border-color:#10b981">'
                    "✓ No conflict edges</div>",
                    unsafe_allow_html=True)

            seq = [col_res[v] for v in sorted(col_res.keys())]
            st.markdown("**Coloring Sequence (by vertex index):**")
            st.code(str(seq), language=None)

            colors_hist: dict = defaultdict(list)
            for v, c in sorted(col_res.items()):
                colors_hist[c].append(v)
            st.markdown("**Vertices grouped by color:**")
            for c, verts in sorted(colors_hist.items()):
                st.markdown(
                    f'<div class="algo-step" '
                    f'style="border-color:{PALETTE[c % len(PALETTE)]}">'
                    f'<b style="color:{PALETTE[c % len(PALETTE)]}">'
                    f"{COLOR_NAMES[c % len(COLOR_NAMES)]}</b>: vertices {verts}</div>",
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Partition Views
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    if not st.session_state.partition_figs:
        st.info("Run the algorithm first to see partition views.")
    else:
        st.markdown("### Louvain Partition Views")
        st.caption(
            "Each panel shows the Louvain community partition used for a given k, "
            "with intra-community edges (solid) and cross-community edges (dashed red), "
            "plus the interaction graph of supernodes."
        )
        part_figs = st.session_state.partition_figs
        for k_val in sorted(part_figs.keys(), reverse=True):
            fig = part_figs[k_val]
            st.markdown(f"**k = {k_val}**")
            st.pyplot(fig, use_container_width=True)
            img_bytes = _fig_to_bytes(fig)
            st.download_button(
                f"⬇️ Download k={k_val} partition PNG", img_bytes,
                file_name=f"partition_k{k_val}.png",
                mime="image/png",
                key=f"dl_part_{k_val}",
            )
            st.markdown("---")