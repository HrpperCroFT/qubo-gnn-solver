from pathlib import Path

import networkx as nx


def load_mwis_instance(instance_dir):
    """
    Load an MWIS instance from a directory.

    Expected files:

        conflict_graph.txt  — first line: "n m", then m lines "u v" (1-indexed)
        node_weights.txt    — lines "vertex weight" (1-indexed)

    Returns (nx_graph, weights, edges, n).
    """
    instance_dir = Path(instance_dir)
    graph_file = instance_dir / "conflict_graph.txt"
    weights_file = instance_dir / "node_weights.txt"

    if not graph_file.exists():
        raise FileNotFoundError(graph_file)
    if not weights_file.exists():
        raise FileNotFoundError(weights_file)

    with open(graph_file) as f:
        first = f.readline().strip()
        if not first:
            raise RuntimeError(f"Empty graph file: {graph_file}")
        n, m = map(int, first.split())
        edges = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = map(int, line.split())
            edges.append((u - 1, v - 1))

    weights = [0.0] * n
    with open(weights_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            vertex = int(parts[0])
            weight = float(parts[1])
            weights[vertex - 1] = weight

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(n))
    nx_graph.add_edges_from(edges)
    for i, w in enumerate(weights):
        nx_graph.nodes[i]["weight"] = w

    print(
        f"Loaded graph: nodes={n} edges={len(edges)}"
        f" weight_range=[{min(weights):.2f}, {max(weights):.2f}]"
    )
    return nx_graph, weights, edges, n


def load_reference_solution(instance_dir, n):
    """Read solution.txt if present. Returns set[int] (0-indexed) or None."""
    path = Path(instance_dir) / "solution.txt"
    if not path.exists():
        return None
    vertices = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                vertices.add(int(line) - 1)
    return vertices


def reference_weight(reference_vertices, weights):
    if reference_vertices is None:
        return None
    return sum(weights[v] for v in reference_vertices)
