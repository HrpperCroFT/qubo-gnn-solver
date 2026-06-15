"""
High-level solver for a single MWIS instance and a full dataset directory.
"""

import json
from itertools import chain
from pathlib import Path

import dgl
import torch

from ..models import build_model
from ..utils import set_seed
from .config import (
    TORCH_DEVICE, TORCH_DTYPE,
    NUMBER_EPOCHS, LEARNING_RATE, PROB_THRESHOLD,
    TOL, PATIENCE, DIM_EMBEDDING, HIDDEN_DIM,
    DROPOUT, NUMBER_CLASSES, PENALTY_COEFF, LOG_EVERY,
)
from .instance_loader import load_mwis_instance, load_reference_solution, reference_weight
from .logging_utils import InstanceLogger
from .qubo import wmis_to_qubo, verify_wmis, selected_vertices_from_bitstring
from .training import run_gnn_training


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_output_solution(instance_dir, selected_vertices, weight_sum, runtime_seconds, number_epochs):
    path = Path(instance_dir) / f"output_gnn_epochs_{number_epochs}.txt"
    with open(path, "w") as f:
        f.write(f"Weight: {weight_sum}, time: {runtime_seconds:.2f}s\n")
        f.write(" ".join(str(v) for v in selected_vertices) + "\n")
    return path


def _save_validity_file(instance_dir, is_valid, weight_sum, selected_count, runtime_seconds):
    path = Path(instance_dir) / "solution_validity.txt"
    with open(path, "w") as f:
        f.write(f"valid={is_valid}\n")
        f.write(f"weight={weight_sum}\n")
        f.write(f"selected_vertices={selected_count}\n")
        f.write(f"runtime_seconds={runtime_seconds:.6f}\n")
    return path


def _save_json_summary(
    instance_dir, is_valid, weight_sum, selected_count, runtime_seconds,
    best_epoch, best_loss, best_score, number_epochs, reference_weight_value=None,
):
    path = Path(instance_dir) / "solution.json"
    payload = {
        "valid": bool(is_valid),
        "weight": float(weight_sum),
        "selected_vertices": int(selected_count),
        "runtime_seconds": float(runtime_seconds),
        "epochs": int(number_epochs),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "best_score": float(best_score),
    }
    if reference_weight_value is not None:
        payload["reference_weight"] = float(reference_weight_value)
        payload["relative_to_reference_percent"] = (
            100.0 * weight_sum / reference_weight_value
        )
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)
    return path


# ---------------------------------------------------------------------------
# Core solve function
# ---------------------------------------------------------------------------

def solve_instance(
    instance_dir,
    *,
    number_epochs=None,
    learning_rate=None,
    penalty_coeff=None,
):
    """
    Solve a single MWIS instance stored in *instance_dir*.

    Parameters default to the values in mwis/config.py when None.
    """
    # Resolve overrides
    number_epochs  = number_epochs  if number_epochs  is not None else NUMBER_EPOCHS
    learning_rate  = learning_rate  if learning_rate  is not None else LEARNING_RATE
    penalty_coeff  = penalty_coeff  if penalty_coeff  is not None else PENALTY_COEFF

    instance_dir = Path(instance_dir)
    print(f"Solving instance: {instance_dir.name}")
    print(
        f"  epochs={number_epochs}  lr={learning_rate}  penalty_coeff={penalty_coeff}"
    )

    set_seed()

    nx_graph, weights, edges, n = load_mwis_instance(instance_dir)

    logger = InstanceLogger(instance_dir / "gnn_training.log")
    logger.log_header(instance_name=instance_dir.name, nodes=n, edges=len(edges))

    # QUBO matrix
    q_torch = wmis_to_qubo(n=n, edges=edges, weights=weights, penalty_coeff=penalty_coeff)
    q_torch = q_torch.to(TORCH_DEVICE).to(TORCH_DTYPE)

    # DGL graph
    graph_dgl = dgl.from_networkx(nx_graph).to(TORCH_DEVICE)

    # Model + optimizer
    net, embed = build_model(
        n_nodes=n,
        dim_embedding=DIM_EMBEDDING,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        number_classes=NUMBER_CLASSES,
        device=TORCH_DEVICE,
        dtype=TORCH_DTYPE,
        extra_input_feats=1,  # vertex weight (diag of Q) appended to features
    )
    optimizer = torch.optim.AdamW(
        chain(net.parameters(), embed.parameters()),
        lr=learning_rate,
    )

    print(f"Starting GNN training for {instance_dir.name}")
    logger.log("Training started")

    result = run_gnn_training(
        q_torch=q_torch,
        dgl_graph=graph_dgl,
        net=net,
        embed=embed,
        optimizer=optimizer,
        number_epochs=number_epochs,
        tol=TOL,
        patience=PATIENCE,
        prob_threshold=PROB_THRESHOLD,
        dim_embedding=DIM_EMBEDDING,
        logger=logger,
        log_every=LOG_EVERY,
    )

    # Extract solution
    bit_list = result["best_bitstring"].detach().cpu().numpy().astype(int).tolist()
    is_valid, weight_sum = verify_wmis(n=n, edges=edges, weights=weights, bitstring=bit_list)
    selected_vertices = selected_vertices_from_bitstring(bit_list)
    selected_count = len(selected_vertices)

    # Compare with reference
    ref_verts = load_reference_solution(instance_dir, n)
    ref_w = reference_weight(ref_verts, weights)

    print(f"\nValid: {is_valid}  Weight: {weight_sum}  Selected: {selected_count}")
    if ref_w is not None:
        print(f"Reference weight: {ref_w}  Relative: {100.0 * weight_sum / ref_w:.2f}%")

    # Save outputs
    _save_output_solution(
        instance_dir, selected_vertices, weight_sum,
        result["training_time"], number_epochs,
    )
    _save_validity_file(
        instance_dir, is_valid, weight_sum, selected_count, result["training_time"],
    )
    _save_json_summary(
        instance_dir, is_valid, weight_sum, selected_count,
        result["training_time"], result["best_epoch"],
        result["best_loss"], result["best_score"],
        number_epochs=number_epochs,
        reference_weight_value=ref_w,
    )
    logger.log_final(valid=is_valid, weight=weight_sum, selected_vertices=selected_count)

    print(f"\nFinished {instance_dir.name}")
    return {
        "instance": instance_dir.name,
        "valid": is_valid,
        "weight": weight_sum,
        "selected_vertices": selected_count,
        "training_time": result["training_time"],
        "best_epoch": result["best_epoch"],
        "best_loss": result["best_loss"],
    }


def run_dataset(
    dataset_dir,
    *,
    number_epochs=None,
    learning_rate=None,
    penalty_coeff=None,
):
    """Run the MWIS solver on every instance subdirectory inside *dataset_dir*."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    instances = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    print(f"Found {len(instances)} instances in {dataset_dir}")

    overrides = dict(
        number_epochs=number_epochs,
        learning_rate=learning_rate,
        penalty_coeff=penalty_coeff,
    )

    success = failed = 0
    for idx, instance_dir in enumerate(instances, start=1):
        print()
        print("=" * 80)
        print(f"[{idx}/{len(instances)}] Processing {instance_dir.name}")
        print("=" * 80)
        try:
            solve_instance(instance_dir, **overrides)
            success += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED: {instance_dir.name}  —  {exc}")

    print()
    print("=" * 80)
    print(f"DONE  successful={success}  failed={failed}")
    print("=" * 80)
