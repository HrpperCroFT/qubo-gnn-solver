"""
QUBO formulation for the (unweighted) Set Cover problem.

Slack variables use the binary (log-trick) encoding by default, which
requires O(N + n * ceil(log2(M))) variables instead of O(N + n*M).
"""

import math

import torch


def build_qubo(n, subsets, A=4.0, B=1.0, use_log_trick=True):
    """
    Build the QUBO matrix for Set Cover with unit costs.

    Parameters
    ----------
    n          : number of elements to cover
    subsets    : list of sets; subsets[j] contains the (1-indexed) elements
                 covered by subset j
    A          : penalty coefficient for coverage constraints
    B          : cost coefficient (all subsets have cost B=1)
    use_log_trick : if True use binary slack encoding; otherwise one-hot

    Returns a float32 CPU tensor of shape (total_vars, total_vars).
    """
    N = len(subsets)

    # rev_subsets[alpha] = set of subset indices that cover element (alpha+1)
    rev_subsets = [set() for _ in range(n)]
    for idx, s in enumerate(subsets):
        for el in s:
            rev_subsets[el - 1].add(idx)

    M = max((len(rev_subsets[el]) for el in range(n)), default=0)

    if use_log_trick:
        k = 1 if M <= 1 else math.floor(math.log2(M)) + 1
        vars_per_element = k
    else:
        k = M
        vars_per_element = M

    total_vars = N + n * vars_per_element
    Q = torch.zeros((total_vars, total_vars))

    # Objective: minimise number of selected subsets
    for i in range(N):
        Q[i, i] += B

    # Coverage constraints
    for alpha in range(n):
        sigma = list(rev_subsets[alpha])
        if not sigma:
            continue

        if use_log_trick:
            for m_idx in range(k):
                i_am = N + alpha * k + m_idx
                Q[i_am, i_am] += A * (pow(2, 2 * m_idx) + 2 * pow(2, m_idx))
                for mp_idx in range(m_idx + 1, k):
                    i_amp = N + alpha * k + mp_idx
                    coeff = 2 * A * pow(2, m_idx + mp_idx)
                    Q[i_am, i_amp] += coeff
                    Q[i_amp, i_am] += coeff
                for i in sigma:
                    Q[i, i_am] -= 2 * A * pow(2, m_idx)
                    Q[i_am, i] -= 2 * A * pow(2, m_idx)
            for i in sigma:
                Q[i, i] -= A
                for j in sigma:
                    if i < j:
                        Q[i, j] += 2 * A
                        Q[j, i] += 2 * A
        else:
            for m_idx in range(1, M + 1):
                i_am = N + alpha * M + (m_idx - 1)
                Q[i_am, i_am] += A * m_idx * m_idx
                for mp_idx in range(m_idx + 1, M + 1):
                    i_amp = N + alpha * M + (mp_idx - 1)
                    coeff = 2 * A * m_idx * mp_idx
                    Q[i_am, i_amp] += coeff
                    Q[i_amp, i_am] += coeff
                for i in sigma:
                    Q[i, i_am] -= 2 * A * m_idx
                    Q[i_am, i] -= 2 * A * m_idx
            for i in sigma:
                Q[i, i] += A
                for j in sigma:
                    if i < j:
                        Q[i, j] += 2 * A
                        Q[j, i] += 2 * A

    return Q
