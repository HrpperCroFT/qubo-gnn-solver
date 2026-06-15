from time import time

import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils import pagerank


def loss_func(probs, Q_mat, epoch=0):
    """
    QUBO energy + annealed non-binary penalty.

    The off-diagonal part of Q is scaled by lambda = epoch / 5000
    to gradually enforce the adjacency constraint.
    """
    penalty_matrix = Q_mat - torch.diag(Q_mat.diag())
    probs_col = probs.unsqueeze(1)
    lbd = epoch / 5e3
    penalty = (1.0 - probs) * probs
    cost = (
        (probs_col.T @ (Q_mat + lbd * penalty_matrix) @ probs_col).squeeze()
        + lbd * penalty.sum()
    )
    return cost


def run_gnn_training(
    q_torch,
    dgl_graph,
    net,
    embed,
    optimizer,
    number_epochs,
    tol,
    patience,
    prob_threshold,
    dim_embedding,
    logger=None,
    log_every=1000,
):
    """
    Train the GNN and return the best binary solution found.

    Returns a dict with keys:
        best_epoch, best_probs, best_bitstring, best_loss,
        final_probs, final_bitstring, inputs, training_time, best_score
    """
    # Edge weights from the off-diagonal QUBO entries
    edge_weight = q_torch - torch.diag(q_torch.diag())
    edge_weight = (edge_weight + edge_weight.T) / 2.0
    edge_weight = edge_weight[dgl_graph.edges()[0], dgl_graph.edges()[1]]

    # Fixed node features: random init + PageRank (computed once)
    inputs = torch.rand(
        (dgl_graph.number_of_nodes(), dim_embedding),
        dtype=q_torch.dtype,
    ).to(q_torch.device)

    walk_features = pagerank(
        dgl_graph.cpu().to_networkx(), 2 * dim_embedding
    ).to(q_torch.device)

    inputs = torch.cat(
        [
            inputs,
            walk_features,
            q_torch.diag().reshape(inputs.shape[0], 1)
        ],
        dim=1
    )

    h0 = torch.zeros(dgl_graph.number_of_nodes(), 1).to(q_torch.device)

    prev_loss = 1.0
    count = 0
    best_epoch = 0
    best_score = float("-inf")
    best_loss = float("inf")
    best_probs = None
    best_bitstring = torch.zeros(dgl_graph.number_of_nodes()).to(q_torch.device)

    start_time = time()
    progress_bar = tqdm(range(number_epochs), desc="Training", leave=True)

    for epoch in progress_bar:
        probs, h0 = net(dgl_graph, inputs, h0.detach())
        probs = probs.squeeze()
        loss = loss_func(probs, q_torch, epoch)
        loss_value = loss.detach().item()

        bitstring = (probs.detach() >= prob_threshold).int()

        if loss_value < best_loss:
            score = (-loss_func(bitstring.float(), q_torch)).item()
            if score > best_score:
                best_loss = loss_value
                best_score = score
                best_epoch = epoch
                best_probs = probs.detach().clone()
                best_bitstring = bitstring.detach().clone()

        if logger is not None and epoch % log_every == 0:
            logger.log_epoch(epoch=epoch, loss=loss_value,
                             best_loss=best_loss, best_score=best_score)

        if epoch % 100 == 0:
            progress_bar.set_postfix(
                loss=f"{loss_value:.5f}",
                best_loss=f"{best_loss:.5f}",
                best_score=f"{best_score:.2f}",
                best_epoch=best_epoch,
            )

        if abs(loss_value - prev_loss) <= tol or loss_value > prev_loss:
            count += 1
        else:
            count = 0

        # Early stopping is intentionally disabled (run full training).
        # if count >= patience:
        #     break

        prev_loss = loss_value
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()

    training_time = time() - start_time
    final_bitstring = (probs.detach() >= prob_threshold).int()

    print(f"GNN training (nodes={dgl_graph.number_of_nodes()}) took {training_time:.2f}s")
    print(f"Best epoch: {best_epoch}  Best loss: {best_loss:.6f}")

    return {
        "best_epoch": best_epoch,
        "best_probs": best_probs,
        "best_bitstring": best_bitstring,
        "best_loss": best_loss,
        "final_probs": probs.detach(),
        "final_bitstring": final_bitstring,
        "inputs": inputs,
        "training_time": training_time,
        "best_score": best_score,
    }
