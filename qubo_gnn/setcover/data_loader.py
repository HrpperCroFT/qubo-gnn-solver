"""
Load Set Cover instances in OR-Library format (scp* and rail* files).
"""

import os


def load_scp_instance(filepath):
    """
    Load a Set Cover instance from an OR-Library file.

    The file type is detected from the basename:
        scp*  — standard SCP format
        rail* — RAIL format

    Returns (n_elements, subsets) where subsets is a list of sets of
    element indices (1-indexed).
    """
    basename = os.path.basename(filepath)
    if basename.startswith("scp"):
        return _load_scp_format(filepath)
    elif basename.startswith("rail"):
        return _load_rail_format(filepath)
    else:
        raise ValueError(
            f"Unknown instance format. "
            f"Filename must start with 'scp' or 'rail', got: {basename!r}"
        )


def _load_scp_format(filepath):
    with open(filepath) as f:
        line = _next_nonempty(f)
        m, n = map(int, line.split())
        _next_nonempty(f)  # cost line (ignored)
        subsets = [set() for _ in range(n)]
        for i in range(1, m + 1):
            parts = list(map(int, f.readline().split()))
            if not parts:
                continue
            k = parts[0]
            for col in parts[1 : 1 + k]:
                subsets[col - 1].add(i)
    return m, subsets


def _load_rail_format(filepath):
    with open(filepath) as f:
        line = _next_nonempty(f)
        m, n = map(int, line.split())
        subsets = [set() for _ in range(n)]
        for j in range(n):
            parts = list(map(int, f.readline().split()))
            if not parts:
                continue
            # cost = parts[0]  (ignored)
            k = parts[1]
            subsets[j] = set(parts[2 : 2 + k])
    return m, subsets


def _next_nonempty(f):
    for line in f:
        line = line.strip()
        if line:
            return line
    raise EOFError("Unexpected end of file")
