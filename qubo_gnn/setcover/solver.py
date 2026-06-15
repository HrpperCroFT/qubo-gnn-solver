"""
High-level solver for a single Set Cover instance and a batch directory.
"""

import os
import time
from itertools import chain
from pathlib import Path

import torch

from ..models import build_model
from .config import (
    TORCH_DEVICE, TORCH_DTYPE,
    N_EPOCHS, LEARNING_RATE, PROB_THRESHOLD,
    TOL, PATIENCE, DIM_EMBEDDING, HIDDEN_DIM,
    DROPOUT, NUMBER_CLASSES, A, B, USE_LOG_TRICK,
)
from .data_loader import load_scp_instance
from .graph_utils import qubo_to_graph, nx_to_dgl
from .qubo import build_qubo
from .training import run_training
from .utils import decode_solution, greedy_cover, verify_set_cover


def solve_instance(
    instance_path,
    output_dir=None,
    *,
    n_epochs=None,
    learning_rate=None,
    penalty_a=None,
    penalty_b=None,
):
    """
    Solve a single Set Cover instance and write .gnnsol and .log files.

    Parameters default to the values in setcover/config.py when None.
    """
    # Resolve overrides
    n_epochs      = n_epochs      if n_epochs      is not None else N_EPOCHS
    learning_rate = learning_rate if learning_rate is not None else LEARNING_RATE
    penalty_a     = penalty_a     if penalty_a     is not None else A
    penalty_b     = penalty_b     if penalty_b     is not None else B

    instance_path = str(instance_path)

    # Default output dir: same directory as the instance file
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(instance_path))
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load instance
    n_elements, subsets = load_scp_instance(instance_path)
    n_subsets = len(subsets)
    print(
        f"Loaded {instance_path}: {n_elements} elements, {n_subsets} subsets  "
        f"[epochs={n_epochs}, lr={learning_rate}, A={penalty_a}, B={penalty_b}]"
    )

    # Build QUBO
    Q = build_qubo(n_elements, subsets, A=penalty_a, B=penalty_b, use_log_trick=USE_LOG_TRICK)
    Q = Q.to(TORCH_DEVICE).type(TORCH_DTYPE)

    # Build graph
    nx_graph = qubo_to_graph(Q)
    dgl_graph = nx_to_dgl(nx_graph, TORCH_DEVICE)
    n_qubits = nx_graph.number_of_nodes()

    # Model + optimizer
    net, embed = build_model(
        n_nodes=n_qubits,
        dim_embedding=DIM_EMBEDDING,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        number_classes=NUMBER_CLASSES,
        device=TORCH_DEVICE,
        dtype=TORCH_DTYPE,
    )
    optimizer = torch.optim.Adam(
        chain(net.parameters(), embed.parameters()),
        lr=learning_rate,
    )

    # Train
    start_time = time.time()
    best_bitstring, log_lines, train_time = run_training(
        Q, dgl_graph, net, embed, optimizer,
        n_epochs, TOL, PATIENCE, PROB_THRESHOLD,
        penalty_a, n_elements, DIM_EMBEDDING, TORCH_DEVICE,
    )
    total_time = time.time() - start_time

    # Decode and validate
    subset_bits = decode_solution(best_bitstring, n_subsets)
    is_valid, num_selected = verify_set_cover(n_elements, subsets, subset_bits)

    if not is_valid:
        subset_bits = greedy_cover(n_elements, subsets, subset_bits)
        is_valid, num_selected = verify_set_cover(n_elements, subsets, subset_bits)

    # Write outputs
    base = os.path.splitext(os.path.basename(instance_path))[0]
    sol_path = os.path.join(output_dir, base + ".gnnsol")
    log_path = os.path.join(output_dir, base + ".log")

    selected_indices = sorted(i + 1 for i, bit in enumerate(subset_bits) if bit == 1)
    with open(sol_path, "w") as f:
        f.write(f"{num_selected} {is_valid} {total_time:.6f}\n")
        f.write(" ".join(map(str, selected_indices)) + "\n")

    with open(log_path, "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    print(f"Finished {instance_path}: selected={num_selected}, time={total_time:.3f}s")


def run_dataset(
    input_path,
    output_dir=None,
    *,
    n_epochs=None,
    learning_rate=None,
    penalty_a=None,
    penalty_b=None,
):
    """
    Run the Set Cover solver on every scp*/rail* file in *input_path*.
    """
    if not os.path.isdir(input_path):
        raise NotADirectoryError(
            f"Expected a directory in dataset mode, got: {input_path}"
        )

    out = output_dir if output_dir else input_path
    os.makedirs(out, exist_ok=True)

    fnames = sorted(
        f for f in os.listdir(input_path)
        if (f.startswith("scp") or f.startswith("rail")) and f.endswith(".txt")
    )
    print(f"Found {len(fnames)} instance(s) in {input_path}")

    overrides = dict(
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        penalty_a=penalty_a,
        penalty_b=penalty_b,
    )

    for fname in fnames:
        solve_instance(os.path.join(input_path, fname), output_dir=out, **overrides)
