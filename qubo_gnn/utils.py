"""
Shared utilities: PageRank node features and reproducibility seed.
"""

import random

import networkx as nx
import numpy as np
import torch


def pagerank(nx_graph, feature_dim=10):
    """
    Compute PageRank for every node and broadcast the scalar value into a
    feature vector of length *feature_dim*.

    Returns a float32 tensor of shape (n_nodes, feature_dim).
    """
    n_nodes = nx_graph.number_of_nodes()
    features = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)
    pr = nx.pagerank(nx.Graph(nx_graph))
    for node, value in pr.items():
        features[node, :] = value
    return features


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
