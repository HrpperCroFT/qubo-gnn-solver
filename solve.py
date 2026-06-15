#!/usr/bin/env python3
"""
Unified GNN-based QUBO solver for MWIS and Set Cover.

Usage — single instance
-----------------------
  python solve.py mwis     <instance_dir>
  python solve.py setcover <instance_file>

Usage — full dataset
--------------------
  python solve.py mwis     <dataset_dir>  --dataset
  python solve.py setcover <instances_dir> --dataset

Override hyperparameters (both problems)
-----------------------------------------
  --epochs N        number of training epochs
  --lr LR           learning rate

MWIS-specific overrides
------------------------
  --penalty-coeff C   QUBO penalty coefficient (must be > 1)

Set Cover-specific overrides
-----------------------------
  --penalty-a A       constraint-violation penalty (must be > penalty-b)
  --penalty-b B       objective cost weight        (default 1.0)
  --output-dir DIR    where to write results (default: input directory)
"""

import argparse
import sys

from qubo_gnn.mwis.config import (
    NUMBER_EPOCHS as MWIS_EPOCHS,
    LEARNING_RATE as MWIS_LR,
    PENALTY_COEFF as MWIS_PENALTY_COEFF,
)
from qubo_gnn.setcover.config import (
    N_EPOCHS as SC_EPOCHS,
    LEARNING_RATE as SC_LR,
    A as SC_A,
    B as SC_B,
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_mwis_params(args):
    if args.penalty_coeff is not None and args.penalty_coeff <= 1:
        sys.exit(
            f"error: --penalty-coeff must be > 1, got {args.penalty_coeff}"
        )
    if args.lr is not None and args.lr <= 0:
        sys.exit(f"error: --lr must be positive, got {args.lr}")
    if args.epochs is not None and args.epochs < 1:
        sys.exit(f"error: --epochs must be >= 1, got {args.epochs}")


def _check_sc_params(args):
    a = args.penalty_a if args.penalty_a is not None else SC_A
    b = args.penalty_b if args.penalty_b is not None else SC_B
    if a <= b:
        sys.exit(
            f"error: --penalty-a ({a}) must be strictly greater than --penalty-b ({b})"
        )
    if args.lr is not None and args.lr <= 0:
        sys.exit(f"error: --lr must be positive, got {args.lr}")
    if args.epochs is not None and args.epochs < 1:
        sys.exit(f"error: --epochs must be >= 1, got {args.epochs}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="GNN-based QUBO solver — supports MWIS and Set Cover",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="problem", required=True)

    # ---- shared arguments factory ----------------------------------------
    def add_shared_args(p):
        p.add_argument(
            "path",
            help="Path to a single instance (default) or a dataset directory (with --dataset)",
        )
        p.add_argument(
            "--dataset",
            action="store_true",
            help="Treat <path> as a dataset directory and solve all instances inside it",
        )
        p.add_argument(
            "--epochs",
            type=int,
            default=None,
            metavar="N",
            help="Number of training epochs (default: %(default)s → uses problem default)",
        )
        p.add_argument(
            "--lr",
            type=float,
            default=None,
            metavar="LR",
            help="Learning rate (default: uses problem default)",
        )

    # ---- mwis ---------------------------------------------------------------
    mwis_p = subparsers.add_parser(
        "mwis",
        help="Solve Maximum Weighted Independent Set",
        description=(
            "Single-instance mode: <path> is the instance directory "
            "(must contain conflict_graph.txt, node_weights.txt).\n"
            "Dataset mode (--dataset): <path> contains one instance "
            "subdirectory per instance.\n\n"
            f"Defaults: epochs={MWIS_EPOCHS}, lr={MWIS_LR}, "
            f"penalty-coeff={MWIS_PENALTY_COEFF}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared_args(mwis_p)
    mwis_p.add_argument(
        "--penalty-coeff",
        type=float,
        default=None,
        metavar="C",
        help=f"QUBO penalty coefficient λ = C × max_weight (must be > 1, default {MWIS_PENALTY_COEFF})",
    )

    # ---- setcover -----------------------------------------------------------
    sc_p = subparsers.add_parser(
        "setcover",
        help="Solve Set Cover (OR-Library scp*/rail* format)",
        description=(
            "Single-instance mode: <path> is the instance file "
            "(filename must start with 'scp' or 'rail').\n"
            "Dataset mode (--dataset): <path> is a directory; all "
            "scp*.txt and rail*.txt files inside are solved.\n\n"
            f"Defaults: epochs={SC_EPOCHS}, lr={SC_LR}, "
            f"penalty-a={SC_A}, penalty-b={SC_B}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared_args(sc_p)
    sc_p.add_argument(
        "--penalty-a",
        type=float,
        default=None,
        metavar="A",
        help=f"Constraint-violation penalty coefficient (must be > penalty-b, default {SC_A})",
    )
    sc_p.add_argument(
        "--penalty-b",
        type=float,
        default=None,
        metavar="B",
        help=f"Objective cost weight (default {SC_B})",
    )
    sc_p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory for output files (default: same location as input)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.problem == "mwis":
        _check_mwis_params(args)
        from qubo_gnn.mwis.solver import solve_instance, run_dataset
        overrides = dict(
            number_epochs=args.epochs,
            learning_rate=args.lr,
            penalty_coeff=args.penalty_coeff,
        )
        if args.dataset:
            run_dataset(args.path, **overrides)
        else:
            solve_instance(args.path, **overrides)

    else:  # setcover
        _check_sc_params(args)
        from qubo_gnn.setcover.solver import solve_instance, run_dataset
        overrides = dict(
            n_epochs=args.epochs,
            learning_rate=args.lr,
            penalty_a=args.penalty_a,
            penalty_b=args.penalty_b,
        )
        if args.dataset:
            run_dataset(args.path, output_dir=args.output_dir, **overrides)
        else:
            solve_instance(args.path, output_dir=args.output_dir, **overrides)


if __name__ == "__main__":
    main()
