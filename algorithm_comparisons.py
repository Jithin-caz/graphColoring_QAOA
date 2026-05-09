import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer.primitives import Sampler
from scipy.optimize import minimize

# --- Helper Functions ---

def get_cost(bitstring, G, k, lambda_node=10, lambda_edge=1):
    """Calculate the classical cost of a given bitstring for the GCP."""
    n = G.number_of_nodes()
    cost = 0
    bits = [int(b) for b in reversed(bitstring)] # qiskit endianness
    
    # Node penalty: each node must have exactly 1 color
    for v in range(n):
        node_sum = sum(bits[v*k + c] for c in range(k))
        cost += lambda_node * (node_sum - 1)**2
        
    # Edge penalty: connected nodes shouldn't share the same color
    for u, v in G.edges():
        for c in range(k):
            if bits[u*k + c] == 1 and bits[v*k + c] == 1:
                cost += lambda_edge
    return cost

def evaluate_circuit(qc, G, k, maxiter=30):
    """Optimize the parameterized circuit and return the best bitstring and cost."""
    sampler = Sampler()
    num_params = qc.num_parameters
    
    best_overall_cost = float('inf')
    best_overall_bitstring = None
    
    def objective(params):
        nonlocal best_overall_cost, best_overall_bitstring
        bound_qc = qc.assign_parameters(params)
        result = sampler.run([bound_qc], shots=512).result()
        counts = result.quasi_dists[0]
        expected_cost = 0
        total_prob = 0
        for state_int, prob in counts.items():
            bitstring = format(state_int, f'0{qc.num_qubits}b')
            cost = get_cost(bitstring, G, k)
            expected_cost += cost * prob
            total_prob += prob
            
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_bitstring = bitstring
                
        return expected_cost / total_prob if total_prob > 0 else 0

    if num_params > 0:
        initial_params = np.random.uniform(0, np.pi, num_params)
        res = minimize(objective, initial_params, method='COBYLA', options={'maxiter': maxiter})
        best_params = res.x
    else:
        best_params = []
        
    bound_qc = qc.assign_parameters(best_params)
    result = sampler.run([bound_qc], shots=1024).result()
    counts = result.quasi_dists[0]
    
    for state_int, prob in counts.items():
        bitstring = format(state_int, f'0{qc.num_qubits}b')
        cost = get_cost(bitstring, G, k)
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_bitstring = bitstring
            
    return best_overall_bitstring, best_overall_cost

def decode_one_hot(bitstring, G, k):
    """Decode a one-hot bitstring into a graph coloring and calculate metrics."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    bits = [int(b) for b in reversed(bitstring)]
    coloring = {}
    valid_one_hot = True
    
    for i, v in enumerate(nodes):
        node_bits = bits[i*k:(i+1)*k]
        if sum(node_bits) != 1:
            valid_one_hot = False
        try:
            color = node_bits.index(1)
        except ValueError:
            color = 0 # Default if no 1 is found
        coloring[v] = color
        
    conflicts = sum(1 for u, v in G.edges() if coloring.get(u) == coloring.get(v))
    conflict_rate = conflicts / max(G.number_of_edges(), 1)
    is_valid = valid_one_hot and (conflicts == 0)
    
    return coloring, conflicts, conflict_rate, is_valid

def print_result(name, bitstring, cost, G, k):
    coloring, conflicts, conflict_rate, is_valid = decode_one_hot(bitstring, G, k)
    colors_used = len(set(coloring.values()))
    print(f"\n{name}...")
    print(f"           colors used : {colors_used}")
    print(f"           conflict rate: {conflict_rate:.4f}")
    print(f"           valid        : {is_valid}")
    print(f"           coloring     : {coloring}")
    print(f"           raw bitstring: {bitstring} (cost: {cost})")

# --- Algorithms ---

def get_basic_qaoa_circuit(G, k, p, lambda_node=10, lambda_edge=1):
    """Standard Penalty-Based QAOA for GCP."""
    n = G.number_of_nodes()
    nq = n * k
    gamma = ParameterVector('γ', p)
    beta = ParameterVector('β', p)
    qc = QuantumCircuit(nq)
    
    qc.h(range(nq))
    
    for l in range(p):
        # 1. Cost Hamiltonian
        # Node Penalty: lambda_node * (sum(x_vc) - 1)^2
        for v in range(n):
            for c in range(k):
                target = v * k + c
                theta_lin = 2 * gamma[l] * (-lambda_node * (k - 2) / 2)
                if theta_lin != 0: qc.rz(theta_lin, target)
            for c1 in range(k):
                for c2 in range(c1 + 1, k):
                    t1 = v * k + c1; t2 = v * k + c2
                    theta_quad = 2 * gamma[l] * (lambda_node / 2)
                    qc.cx(t1, t2); qc.rz(theta_quad, t2); qc.cx(t1, t2)
        
        # Edge Penalty
        for u, v in G.edges():
            for c in range(k):
                t1 = u * k + c; t2 = v * k + c
                theta_lin = 2 * gamma[l] * (-lambda_edge / 4)
                qc.rz(theta_lin, t1); qc.rz(theta_lin, t2)
                theta_quad = 2 * gamma[l] * (lambda_edge / 4)
                qc.cx(t1, t2); qc.rz(theta_quad, t2); qc.cx(t1, t2)
        
        # 2. Mixing Hamiltonian
        for q in range(nq):
            qc.rx(2 * beta[l], q)
            
    qc.measure_all()
    return qc

def get_constrained_qaoa_circuit(G, k, p):
    """Symmetry-Protected QAOA using XY-Mixer."""
    n = G.number_of_nodes()
    nq = n * k
    gamma = ParameterVector('γ', p)
    beta = ParameterVector('β', p)
    qc = QuantumCircuit(nq)
    
    # W-states
    for v in range(n):
        qc.x(v * k)
        for c in range(k - 1):
            qc.ch(v * k + c, v * k + c + 1)
            qc.cx(v * k + c + 1, v * k + c)

    for l in range(p):
        for u, v in G.edges():
            for c in range(k):
                t1 = u * k + c; t2 = v * k + c
                qc.cx(t1, t2)
                qc.rz(2 * gamma[l], t2)
                qc.cx(t1, t2)
        
        for v in range(n):
            for c in range(k):
                idx1, idx2 = v * k + c, v * k + ((c + 1) % k)
                qc.iswap(idx1, idx2)
                qc.rz(beta[l], idx2)
                qc.iswap(idx1, idx2)
                
    qc.measure_all()
    return qc

def rqaoa_iteration(G, k, p, threshold=2):
    """Recursive QAOA logic: eliminates variables based on Z-Z correlations."""
    if G.number_of_nodes() <= threshold:
        return nx.coloring.greedy_color(G)
    
    qc = get_basic_qaoa_circuit(G, k, p)
    num_params = qc.num_parameters
    sampler = Sampler()
    bound_qc = qc.assign_parameters(np.random.uniform(0, np.pi, num_params))
    res = sampler.run([bound_qc], shots=1024).result()
    counts = res.quasi_dists[0]
    
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    max_corr = -1
    best_pair = (nodes[0], nodes[1]) if n > 1 else (nodes[0], nodes[0])
    
    for i in range(n):
        for j in range(i+1, n):
            u = nodes[i]; v = nodes[j]
            corr = 0
            for c in range(k):
                ex = 0
                for state, prob in counts.items():
                    bitstring = format(state, f'0{qc.num_qubits}b')
                    bits = [int(b) for b in reversed(bitstring)]
                    z_u = 1 - 2*bits[i*k + c]
                    z_v = 1 - 2*bits[j*k + c]
                    ex += z_u * z_v * prob
                corr += abs(ex)
            if corr > max_corr:
                max_corr = corr
                best_pair = (u, v)
                
    reduced_G = G.copy()
    if best_pair[1] in reduced_G:
        reduced_G.remove_node(best_pair[1])
    return reduced_G

def get_warm_started_qaoa(G, k, p, classical_probs):
    """WS-QAOA with classically biased initial state."""
    n = G.number_of_nodes()
    nq = n * k
    gamma = ParameterVector('γ', p)
    beta = ParameterVector('β', p)
    qc = QuantumCircuit(nq)
    
    # Biased Initialization
    for i, p_val in enumerate(classical_probs):
        theta = 2 * np.arcsin(np.sqrt(p_val))
        qc.ry(theta, i)
        
    lambda_node, lambda_edge = 10, 1
    for l in range(p):
        for v in range(n):
            for c in range(k):
                target = v * k + c
                theta_lin = 2 * gamma[l] * (-lambda_node * (k - 2) / 2)
                if theta_lin != 0: qc.rz(theta_lin, target)
            for c1 in range(k):
                for c2 in range(c1 + 1, k):
                    t1 = v * k + c1; t2 = v * k + c2
                    qc.cx(t1, t2); qc.rz(2 * gamma[l] * (lambda_node / 2), t2); qc.cx(t1, t2)
                    
        for u, v in G.edges():
            for c in range(k):
                t1 = u * k + c; t2 = v * k + c
                theta_lin = 2 * gamma[l] * (-lambda_edge / 4)
                qc.rz(theta_lin, t1); qc.rz(theta_lin, t2)
                qc.cx(t1, t2); qc.rz(2 * gamma[l] * (lambda_edge / 4), t2); qc.cx(t1, t2)
        
        # Modified Mixer for WS-QAOA
        for i, p_val in enumerate(classical_probs):
            theta = 2 * np.arcsin(np.sqrt(p_val))
            qc.ry(-theta, i)
            qc.rx(2 * beta[l], i)
            qc.ry(theta, i)
            
    qc.measure_all()
    return qc

def adapt_qaoa_growth(G, k, p):
    """Adaptive operator selection based on gradients (simplified mock)."""
    n = G.number_of_nodes()
    nq = n * k
    qc = QuantumCircuit(nq)
    qc.h(range(nq))
    
    operator_pool = ["RX", "RY", "RZ", "CX"]
    
    for l in range(p):
        # Mock gradient calculation for Adapt-QAOA
        gradients = [np.random.uniform(0, 1) for _ in operator_pool]
        best_op = operator_pool[np.argmax(gradients)]
        
        param = ParameterVector(f'p_{l}', 1)[0]
        if best_op == "RX":
            qc.rx(param, 0)
        elif best_op == "RY":
            qc.ry(param, 0)
        elif best_op == "RZ":
            qc.rz(param, 0)
        elif best_op == "CX" and nq > 1:
            qc.cx(0, 1)
            
    qc.measure_all()
    return qc

def get_multi_angle_qaoa(G, k, p):
    """Multi-Angle QAOA with independent angles per node/edge."""
    n = G.number_of_nodes()
    edges = list(G.edges())
    gamma = ParameterVector('γ', len(edges) * p) 
    beta = ParameterVector('β', n * p)
    qc = QuantumCircuit(n * k)
    qc.h(range(n * k))
    
    for l in range(p):
        for i, (u, v) in enumerate(edges):
            g_idx = l * len(edges) + i
            for c in range(k):
                t1 = u * k + c; t2 = v * k + c
                qc.cx(t1, t2)
                qc.rz(2 * gamma[g_idx], t2)
                qc.cx(t1, t2)
        
        for v in range(n):
            b_idx = l * n + v
            for c in range(k):
                qc.rx(2 * beta[b_idx], v * k + c)
                
    qc.measure_all()
    return qc

# --- Main Execution ---

if __name__ == "__main__":
    print("Initializing test graph (Cycle / C5)...")
    G = nx.cycle_graph(5)
    k = 3 # Colors
    p = 2 # QAOA layers
    
    print(f"Graph nodes: {G.nodes()}, edges: {G.edges()}")
    
    qc_basic = get_basic_qaoa_circuit(G, k, p)
    bitstring, cost = evaluate_circuit(qc_basic, G, k, maxiter=5)
    print_result("1. Testing Basic QAOA", bitstring, cost, G, k)
    
    qc_const = get_constrained_qaoa_circuit(G, k, p)
    bitstring, cost = evaluate_circuit(qc_const, G, k, maxiter=5)
    print_result("2. Testing Constrained QAOA", bitstring, cost, G, k)
    
    print("\n3. Testing RQAOA Iteration...")
    reduced_G = rqaoa_iteration(G, k, p)
    if isinstance(reduced_G, dict):
        print(f"           RQAOA solved classically: {reduced_G}")
    else:
        print(f"           RQAOA reduced graph to {reduced_G.number_of_nodes()} nodes.")
        
    classical_probs = np.random.uniform(0.1, 0.9, G.number_of_nodes() * k)
    qc_ws = get_warm_started_qaoa(G, k, p, classical_probs)
    bitstring, cost = evaluate_circuit(qc_ws, G, k, maxiter=5)
    print_result("4. Testing WS-QAOA", bitstring, cost, G, k)
    
    qc_adapt = adapt_qaoa_growth(G, k, p)
    bitstring, cost = evaluate_circuit(qc_adapt, G, k, maxiter=5)
    print_result("5. Testing Adapt-QAOA", bitstring, cost, G, k)
    
    qc_multi = get_multi_angle_qaoa(G, k, p)
    bitstring, cost = evaluate_circuit(qc_multi, G, k, maxiter=5)
    print_result("6. Testing Multi-Angle QAOA", bitstring, cost, G, k)
    
    print("\nAll algorithms successfully executed!")
