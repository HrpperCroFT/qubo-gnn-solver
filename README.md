# qubo_gnn_solver

GNN-based solver for combinatorial optimisation problems formulated as QUBO
(Quadratic Unconstrained Binary Optimization).  
Supports two problems:

| Problem | Solver | Reference |
|---------|--------|-----------|
| Maximum Weighted Independent Set (MWIS) | `mwis` | Schuetz et al. 2021; Pugacheva et al. 2024 |
| Set Cover (unweighted, OR-Library) | `setcover` | — |

---

## Installation

### 1. Create a Python 3.12 environment

```bash
conda create -n qubo_gnn python=3.12
conda activate qubo_gnn
```

### 2. Install PyTorch

CUDA 12.1 (GPU):
```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

CPU-only:
```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install DGL

DGL must match the PyTorch version. The CUDA 12.1 wheel:
```bash
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
```

CPU-only:
```bash
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.3/cpu/repo.html
```

### 4. Install the remaining dependencies

```bash
pip install networkx==3.6.1 numpy==1.26.4 tqdm==4.67.1
```

> **Note:** `requirements.txt` lists the exact versions for reference, but
> PyTorch and DGL cannot be installed from that file alone because they
> require special index URLs (see steps 2–3 above).

---

## Usage

All commands are run from inside the `qubo_gnn_solver/` directory.

```
python solve.py {mwis,setcover} <path> [--dataset] [options]
```

`<path>` is the instance (default) or dataset directory / file (`--dataset`).

---

### MWIS

```bash
# Single instance
python solve.py mwis <instance_dir>

# Full dataset (directory of instance subdirectories)
python solve.py mwis <dataset_dir> --dataset
```

Each instance directory must contain:

| File | Description |
|------|-------------|
| `conflict_graph.txt` | First line: `n m`; then `m` lines `u v` (1-indexed) |
| `node_weights.txt` | Lines `vertex weight` (1-indexed) |
| `solution.txt` *(optional)* | Reference solution: one 1-indexed vertex per line |

**Output** (written into each instance directory):

| File | Description |
|------|-------------|
| `output_gnn_epochs_<N>.txt` | Selected vertices and weight |
| `solution_validity.txt` | Human-readable validity summary |
| `solution.json` | Machine-readable summary (includes quality vs. reference) |
| `gnn_training.log` | Per-epoch training log |

**Override options:**

| Flag | Default | Constraint |
|------|---------|------------|
| `--epochs N` | 400 000 | ≥ 1 |
| `--lr LR` | 1e-5 | > 0 |
| `--penalty-coeff C` | 2.5 | > 1 |

**Examples:**
```bash
python solve.py mwis /data/instances/alabama-AM2/
python solve.py mwis /data/mwis_dataset/ --dataset
python solve.py mwis /data/mwis_dataset/ --dataset --epochs 100000 --penalty-coeff 3.0
```

---

### Set Cover

```bash
# Single instance
python solve.py setcover <instance_file>

# Full dataset (directory of scp*.txt / rail*.txt files)
python solve.py setcover <instances_dir> --dataset
```

Instance filenames must start with `scp` or `rail` (OR-Library format, `.txt`).

**Output** (per instance):

| File | Description |
|------|-------------|
| `<name>.gnnsol` | `num_selected is_valid total_time` + selected subset indices |
| `<name>.log` | Per-epoch training log |

If the GNN solution is infeasible, greedy post-processing guarantees a valid covering.

**Override options:**

| Flag | Default | Constraint |
|------|---------|------------|
| `--epochs N` | 60 000 | ≥ 1 |
| `--lr LR` | 1e-3 | > 0 |
| `--penalty-a A` | 4.0 | > penalty-b |
| `--penalty-b B` | 1.0 | < penalty-a |
| `--output-dir DIR` | same as input | — |

**Examples:**
```bash
python solve.py setcover /data/setcover/scp41.txt
python solve.py setcover /data/setcover/ --dataset --output-dir /results/
python solve.py setcover /data/setcover/ --dataset --penalty-a 6.0 --penalty-b 1.0

**Output** (per instance):

| File | Description |
|------|-------------|
| `<name>.gnnsol` | `num_selected is_valid total_time` + selected subset indices |
| `<name>.log` | Per-epoch training log |

If the GNN solution is infeasible, greedy post-processing is applied
automatically to guarantee a valid covering.

**Examples:**
```bash
# Single instance
python solve.py setcover /data/setcover/scp41.txt

# All instances in a directory, separate output folder
python solve.py setcover /data/setcover/ --output-dir /results/setcover/
```

---

## Project structure

```
qubo_gnn_solver/
├── solve.py                  # unified entry point
├── requirements.txt
├── README.md
└── qubo_gnn/
    ├── models.py             # shared ResSAGE GNN architecture
    ├── utils.py              # shared PageRank features + seed utility
    ├── mwis/
    │   ├── config.py         # hyperparameters (400 000 epochs, AdamW, lr=1e-5)
    │   ├── instance_loader.py
    │   ├── qubo.py           # MWIS → QUBO reduction
    │   ├── training.py       # training loop with annealed penalty
    │   ├── logging_utils.py
    │   └── solver.py         # solve_instance / run_dataset
    └── setcover/
        ├── config.py         # hyperparameters (60 000 epochs, Adam, lr=1e-3)
        ├── data_loader.py    # OR-Library scp/rail parser
        ├── qubo.py           # Set Cover → QUBO (binary slack encoding)
        ├── graph_utils.py
        ├── training.py       # training loop with early stopping
        ├── utils.py          # verify, decode, greedy post-processing
        └── solver.py         # solve_instance / run_dataset
```

---

## Hyperparameters

### MWIS

| Parameter | Value |
|-----------|-------|
| Epochs | 400 000 |
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| Hidden dim | 32 |
| Embedding dim | 10 |
| Penalty coeff | 2.5 |
| Early stopping | disabled |

### Set Cover

| Parameter | Value |
|-----------|-------|
| Epochs | 60 000 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Hidden dim | 31 |
| Embedding dim | 10 |
| Constraint penalty A | 4.0 |
| Cost weight B | 1.0 |
| Early stopping | patience = 100 |
