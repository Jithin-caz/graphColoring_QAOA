# Efficient Hybrid Variational Quantum Algorithm for Graph Coloring

A faithful Python implementation of the quantum graph coloring algorithm from:

> Liu et al., **"Efficient hybrid variational quantum algorithm for solving graph coloring problem"**, arXiv:2504.21335v1 [quant-ph] (2025)

## Overview

This project implements a hierarchical hybrid quantum-classical approach to solve the **graph k-coloring problem** using **QAOA (Quantum Approximate Optimization Algorithm)** combined with classical optimization techniques. The algorithm is designed for NISQ (Noisy Intermediate-Scale Quantum) devices with limited qubit budgets.

### Key Features

- **Louvain Community Partitioning**: Decomposes large graphs into manageable subgraphs
- **QAOA-based Subgraph Coloring**: Quantum solver for individual subgraph coloring with fixed boundary constraints
- **Interaction Graph Coloring**: Classical greedy coloring of the compressed interaction graph
- **Feedback Correction**: Iterative refinement with dynamic constraint updates
- **Isomorphic Subgraph Reuse**: Caches and reuses solutions for structurally identical subgraphs (Eq. 5)
- **Conflict Resolution**: Greedy conflict-correction for final validation

## Installation

### Requirements
- Python 3.10 or 3.11 (3.13 not yet supported by projectq)
- Qiskit & Qiskit-Aer (or Qiskit + MindQuantum)
- NetworkX, SciPy, NumPy

### Setup

1. **Downgrade Python to 3.10 or 3.11** (if using 3.13)
   ```bash
   # Uninstall Python 3.13 from Control Panel
   # Download and install Python 3.10 or 3.11
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .dj_env
   .dj_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install networkx python-louvain qiskit qiskit-aer scipy numpy
   ```

   **Alternative (if using MindQuantum):**
   ```bash
   pip install networkx python-louvain mindquantum scipy numpy
   ```

## Usage

### Quick Start

```python
import networkx as nx
from main import quantum_graph_coloring

# Create a random graph
G = nx.gnm_random_graph(10, 12, seed=0)

# Solve using the hybrid algorithm
coloring, k_used, conflict_rate = quantum_graph_coloring(
    G,
    k_max=4,           # Maximum colors to try
    Q_available=20,    # Qubit budget
    p=2,               # QAOA circuit depth
    shots=1024,        # Measurement shots
    seed=42
)

print(f"Coloring: {coloring}")
print(f"Colors used: {k_used}")
print(f"Conflict rate: {conflict_rate:.4f}")
```

### Run Examples

```bash
python main.py
```

This runs three example graphs:
1. **10-node, 12-edge** (matches paper dataset)
2. **10-node, 16-edge** (intermediate difficulty)
3. **20-node, 20-edge** (paper dataset)

## Algorithm Pipeline (Fig. 1)

### Step (a): Louvain Partitioning
- Detects communities using modularity optimization
- Limits subgraph size to fit qubit budget: $V_{sub} \leq \lfloor Q / \lceil\log_2(k)\rceil \rfloor$ (Eq. 1)
- Recursively splits oversized communities

### Step (b): QAOA on Subgraphs
- Constructs problem Hamiltonian with edge and vertex constraints (Eqs. 10-12)
- Builds p-layer QAOA ansatz with parameterized cost and mixer layers
- Optimizes using COBYLA with early stopping
- Decodes measurement outcomes to k-colorings

### Step (c): Feedback Correction
- Feeds interaction-graph coloring as hard constraints back to subgraph solver
- Implements isomorphic subgraph detection (Eq. 5) with solution caching
- Iteratively refines until no conflicts remain

### Step (d): Merge & Conflict Resolution
- Merges subgraph colorings using color synthesis (Eq. 6)
- Applies greedy conflict-correction (Eq. 7) if needed
- Validates final coloring

## Key Equations

| Equation | Description |
|----------|-------------|
| **(1)** | Qubit allocation: $V_{sub} \leq \lfloor Q / \lceil\log_2(k)\rceil \rfloor$ |
| **(2)** | Edge conflict function |
| **(4)** | Total cost: $C_{total} = \sum_{edge} + \lambda\sum_{vertex}$ |
| **(6)** | Color merging synthesis |
| **(8)** | Validity: all edges have different colors, fixed constraints respected |
| **(9)** | Conflict rate: $\varepsilon = \|E_{conflict}\| / \|E\|$ |
| **(10)** | Hamiltonian edge term: $H_{edge} = \sum_i (I - Z_u \otimes Z_v)$ |
| **(13)** | Loss function: $\ell(\theta) = \langle\psi(\theta)\|H_C\|\psi(\theta)\rangle$ |

## Parameters

### `quantum_graph_coloring()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `graph` | — | NetworkX graph to color |
| `k_max` | 4 | Maximum colors to sweep (2 to k_max) |
| `Q_available` | 20 | Qubit budget for the quantum processor |
| `p` | 2 | QAOA circuit depth (number of layers) |
| `shots` | 1024 | Number of measurement shots per evaluation |
| `lambda_edge` | 2.0 | Edge conflict penalty weight |
| `feedback_iter` | 5 | Max feedback correction iterations |
| `resolve_iter` | 20 | Max greedy conflict-resolution steps |
| `seed` | 42 | Random seed for reproducibility |

### Returns

```python
(coloring, k_used, conflict_rate)
```

- **coloring**: `Dict[node → color]` — final node-to-color assignment
- **k_used**: `int` — number of colors actually used
- **conflict_rate**: `float` — fraction of conflicting edges ($\varepsilon$)

## Supported Graph Types

- **Random graphs**: `nx.gnm_random_graph(n, m, seed=...)`
- **Cycle graphs**: `nx.cycle_graph(n)`
- **Watts-Strogatz**: `nx.watts_strogatz_graph(n, k, p)`
- **Barabási-Albert**: `nx.barabasi_albert_graph(n, m)`
- **Custom graphs**: Any NetworkX Graph object

## Performance Notes

- **Optimal for**: Graphs with 10–30 nodes on ~20 qubits
- **Circuit depth**: p=1 or p=2 sufficient for most instances
- **Measurement budget**: 512–1024 shots per evaluation recommended
- **Scalability**: Louvain partitioning enables handling of larger graphs

## Troubleshooting

### ImportError: No module named 'projectq'
- Ensure Python 3.10 or 3.11 is installed (not 3.13)
- Upgrade build tools: `pip install --upgrade pip setuptools wheel`

### ImportError: No module named 'qiskit_aer'
```bash
pip install qiskit-aer
```

### Slow execution
- Reduce `shots` parameter (512 instead of 1024)
- Reduce `p` (circuit depth) to 1
- Use smaller graphs for testing

## Citation

If you use this implementation, please cite:

```bibtex
@article{Liu2025QAOA,
  title={Efficient hybrid variational quantum algorithm for solving graph coloring problem},
  author={Liu, et al.},
  journal={arXiv preprint arXiv:2504.21335},
  year={2025}
}
```

## License

This implementation is provided for research and educational purposes.

## References

- Liu et al. (2025) – arXiv:2504.21335v1 [quant-ph]
- Qiskit Documentation: https://qiskit.org/
- NetworkX: https://networkx.org/
- Community Detection (Louvain): https://python-louvain.readthedocs.io/
