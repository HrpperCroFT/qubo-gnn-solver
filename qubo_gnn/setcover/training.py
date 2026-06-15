"""
Training loop for the Set Cover GNN solver.

Key differences from the MWIS loop:
  - The node embedding is re-created each epoch (dynamic input).
  - The loss uses two separate penalty terms (quadratic + entropy) with
    separate annealing schedules.
  - Early stopping is enabled.
"""

import time

import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils import pagerank


def loss_func(probs, Q_mat, epoch):
    """
    QUBO energy + annealed quadratic penalty + annealed entropy penalty.

    lambda_1 = lambda_2 = min(epoch / 500, 1.0)
    """
    probs_ = probs.unsqueeze(1)
    energy = (probs_.T @ Q_mat @ probs_).squeeze()

    penalty_quad = ((probs * (1 - probs)) ** 2).sum()

    eps = 1e-9
    penalty_ent = -(
        probs * torch.log(probs + eps)
        + (1 - probs) * torch.log(1 - probs + eps)
    ).sum()

    lam = min(epoch / 500.0, 1.0)
    return energy + lam * penalty_quad + lam * penalty_ent


def run_training(
    Q,
    dgl_graph,
    net,
    embed,
    optimizer,
    number_epochs,
    tol,
    patience,
    prob_threshold,
    A_qubo,
    n_elements,
    dim_embedding,
    device,
):
    """
    Train the GNN on a Set Cover QUBO and return the best solution found.

    Returns (best_bitstring, log_lines, total_time).
    """
    edge_weight = Q - torch.diag_embed(Q.diag())
    src, dst = dgl_graph.edges()
    edge_weight = edge_weight[src, dst]

    # PageRank features computed once (graph structure doesn't change)
    nx_graph = dgl_graph.cpu().to_networkx()
    walk = pagerank(nx_graph, 2 * dim_embedding).to(device).type(Q.dtype)

    n_nodes = dgl_graph.number_of_nodes()
    node_ids = torch.arange(n_nodes, device=device)

    h0 = torch.zeros(n_nodes, 1, device=device, dtype=Q.dtype)

    prev_loss = 1.0
    count = 0
    best_bitstring = torch.zeros(n_nodes, device=device, dtype=Q.dtype)
    best_loss_val = float("inf")
    best_solution_metric = float("inf")
    best_epoch = 0
    log_lines = []

    start_time = time.time()
    progress_bar = tqdm(range(number_epochs), desc="Training", leave=True)

    for epoch in progress_bar:
        # Embedding is recreated each epoch (dynamic input)
        embed_input = embed(node_ids)
        inputs = torch.cat([embed_input, walk], dim=1)

        probs, h0 = net(dgl_graph, inputs, h0.detach(), edge_weight)
        probs = probs.squeeze()
        loss = loss_func(probs, Q, epoch)
        loss_val = loss.detach().item()

        with torch.no_grad():
            bitstring = (probs.detach() >= prob_threshold).float()
            current_metric = (
                loss_func(bitstring, Q, epoch).item() + A_qubo * n_elements
            )
            if current_metric < best_solution_metric:
                best_solution_metric = current_metric
                best_bitstring = bitstring.clone()
                best_loss_val = loss_val
                best_epoch = epoch

        if epoch % 1000 == 0:
            elapsed = time.time() - start_time
            log_lines.append(
                f"[{elapsed:.2f}s] epoch={epoch} "
                f"loss={loss_val:.8f} best_loss={best_loss_val:.8f} "
                f"best_solution={best_solution_metric:.8f}"
            )

        if epoch % 100 == 0:
            progress_bar.set_postfix(
                loss=f"{loss_val:.5f}",
                best_loss=f"{best_loss_val:.5f}",
                best_score=f"{best_solution_metric:.2f}",
                best_epoch=best_epoch,
            )

        if abs(loss_val - prev_loss) <= tol or loss_val > prev_loss:
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

        prev_loss = loss_val
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()

    total_time = time.time() - start_time
    return best_bitstring, log_lines, total_time
