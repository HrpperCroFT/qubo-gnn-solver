def verify_set_cover(n, subsets, bitstring):
    """
    Check whether the bitstring represents a valid Set Cover.

    Returns (is_valid, num_selected).
    """
    covered = set()
    num_selected = 0
    for i, bit in enumerate(bitstring):
        if bit == 1:
            num_selected += 1
            covered.update(subsets[i])
    return covered == set(range(1, n + 1)), num_selected


def decode_solution(full_bitstring, n_subsets):
    """
    Extract the first n_subsets bits (subset-selection variables) from the
    full solution vector that includes slack variables.
    """
    if hasattr(full_bitstring, "cpu"):
        bits = full_bitstring.cpu().numpy().tolist()
    else:
        bits = list(full_bitstring)
    return bits[:n_subsets]


def greedy_cover(n_elements, subsets, initial_bits=None):
    """
    Greedily add subsets until all elements are covered.

    Starts from *initial_bits* (a GNN-produced partial solution) and
    fills in missing coverage.  Returns a 0/1 list of length len(subsets).
    """
    bits = [0] * len(subsets)
    if initial_bits:
        for i, b in enumerate(initial_bits):
            bits[i] = int(b)

    covered = set()
    for i, b in enumerate(bits):
        if b:
            covered.update(subsets[i])

    universal = set(range(1, n_elements + 1))
    while covered != universal:
        best_idx, best_gain = -1, -1
        for i, s in enumerate(subsets):
            if bits[i] == 0:
                gain = len(s - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
        if best_idx == -1:
            break
        bits[best_idx] = 1
        covered.update(subsets[best_idx])

    return bits
