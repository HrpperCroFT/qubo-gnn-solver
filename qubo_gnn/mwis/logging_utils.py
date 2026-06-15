import time
from pathlib import Path


class InstanceLogger:

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.start_time = time.time()
        with open(self.log_path, "w") as f:
            f.write("==============================\n")
            f.write("MWIS GNN TRAINING LOG\n")
            f.write("==============================\n\n")

    def elapsed_seconds(self):
        return time.time() - self.start_time

    def log(self, message):
        elapsed = self.elapsed_seconds()
        with open(self.log_path, "a") as f:
            f.write(f"[{elapsed:.2f}s] {message}\n")

    def log_epoch(self, epoch, loss, best_loss, best_score):
        elapsed = self.elapsed_seconds()
        line = (
            f"[{elapsed:.2f}s] epoch={epoch} "
            f"loss={loss:.8f} best_loss={best_loss:.8f} "
            f"best_solution={best_score:.8f}"
        )
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def log_header(self, instance_name, nodes, edges):
        self.log(f"instance={instance_name} nodes={nodes} edges={edges}")

    def log_final(self, valid, weight, selected_vertices):
        elapsed = self.elapsed_seconds()
        with open(self.log_path, "a") as f:
            f.write("\n==============================\n")
            f.write("FINAL RESULT\n")
            f.write("==============================\n")
            f.write(f"time={elapsed:.2f}\n")
            f.write(f"valid={valid}\n")
            f.write(f"weight={weight}\n")
            f.write(f"selected_vertices={selected_vertices}\n")
