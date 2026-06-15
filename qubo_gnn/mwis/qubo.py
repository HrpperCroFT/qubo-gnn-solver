import torch


def wmis_to_qubo(n, edges, weights, penalty_coeff=2.0):
    """
    Build the QUBO matrix for MWIS.

    H = -sum(w_i * x_i) + lambda * sum(x_i * x_j)  for each edge (i,j)

    where lambda = penalty_coeff * max(weights).
    """
    if len(weights) != n:
        raise ValueError("len(weights) must equal n")

    max_weight = max(weights) if weights else 1.0
    lam = penalty_coeff * max_weight

    Q = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        Q[i, i] = -weights[i]
    for u, v in edges:
        if u == v:
            continue
        Q[u, v] = lam / 2.0
        Q[v, u] = lam / 2.0
    return Q


def verify_wmis(n, edges, weights, bitstring):
    """Returns (is_valid, total_weight)."""
    for u, v in edges:
        if bitstring[u] == 1 and bitstring[v] == 1:
            total_weight = sum(w for w, b in zip(weights, bitstring) if b)
            return False, total_weight
    total_weight = sum(w for w, b in zip(weights, bitstring) if b)
    return True, total_weight


def bitstring_weight(weights, bitstring):
    return sum(w for w, b in zip(weights, bitstring) if b)


def selected_vertices_from_bitstring(bitstring):
    """Return 1-indexed list of selected vertices."""
    return [i + 1 for i, b in enumerate(bitstring) if b == 1]
