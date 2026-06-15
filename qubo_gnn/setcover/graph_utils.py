import networkx as nx
import torch
import dgl


def qubo_to_graph(Q):
    """Build a NetworkX graph from the non-zero off-diagonal pattern of Q."""
    Q_np = Q.detach().cpu().numpy() if isinstance(Q, torch.Tensor) else Q
    n = Q_np.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = torch.triu(
        torch.tensor(Q_np != 0), diagonal=1
    ).nonzero(as_tuple=True)
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    return G


def nx_to_dgl(nx_graph, device):
    """Convert a NetworkX graph to a DGL graph on the given device."""
    return dgl.from_networkx(nx_graph).to(device)
