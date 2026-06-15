"""
Shared GNN architecture (ResSAGE) used by both MWIS and Set Cover solvers.

The SAGEResBlock here uses BatchNorm(out_channels) — the version with the
fixed batch-norm dimensions from the Set Cover codebase.
"""

from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class SAGEResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, feat_drop=0.0):
        super().__init__()
        self.sage1 = SAGEConv(
            in_channels, out_channels,
            aggregator_type="mean",
            feat_drop=feat_drop,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.sage2 = SAGEConv(
            in_channels, out_channels,
            aggregator_type="pool",
            feat_drop=feat_drop,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, graph, x, edge_weight=None):
        out1 = self.bn1(self.sage1(graph, x, edge_weight))
        out2 = self.bn2(self.sage2(graph, x, edge_weight))
        return self.relu(out1 + out2)


class ResSAGE(nn.Module):

    def __init__(self, in_feats, hidden_sizes, number_classes, dropout, device):
        super().__init__()
        self.dropout_frac = dropout
        self.layers = nn.ModuleList()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        current_dim = in_feats
        for hdim in hidden_sizes:
            self.layers.append(SAGEResBlock(current_dim, hdim).to(device))
            self.layers.append(nn.LeakyReLU())
            current_dim = hdim

        self.layers.append(
            SAGEConv(current_dim, number_classes, aggregator_type="mean").to(device)
        )

    def forward(self, graph, h, h0, edge_weight=None):
        h = torch.cat([h, h0], dim=1)
        blocks = self.layers[:-1][::2]
        activations = self.layers[:-1][1::2]
        for block, act in zip(blocks, activations):
            h = block(graph, h, edge_weight)
            h = act(h)
        h = F.dropout(h, p=self.dropout_frac)
        h0 = self.layers[-1](graph, h, edge_weight)
        probs = torch.sigmoid(h0)
        return probs, h0


def build_model(
    n_nodes, dim_embedding, hidden_dim, dropout, number_classes, device, dtype,
    extra_input_feats=0,
):
    """
    Construct the ResSAGE network and a learnable node embedding table.

    Parameters
    ----------
    extra_input_feats : int
        Additional scalar features concatenated to the standard
        [random_embed (dim_embedding) | pagerank (2*dim_embedding)] input.
        For MWIS this is 1 (the diagonal of Q, i.e. the vertex weight).
        For Set Cover this is 0.

    Returns (net, embed) — the optimizer is created by the caller so that
    each problem can use its own optimizer type and learning rate.
    """
    in_feats = 3 * dim_embedding + number_classes + extra_input_feats
    net = ResSAGE(in_feats, hidden_dim, number_classes, dropout, device)
    net = net.type(dtype).to(device)

    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(dtype).to(device)

    return net, embed
