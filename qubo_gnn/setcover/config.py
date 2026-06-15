import torch

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32

N_EPOCHS = 60000
LEARNING_RATE = 1e-3
PROB_THRESHOLD = 0.5
TOL = 1e-4
PATIENCE = 100
DIM_EMBEDDING = 10
HIDDEN_DIM = 31
DROPOUT = 0.5
NUMBER_CLASSES = 1

# QUBO penalty / cost scaling
A = 4.0   # constraint penalty
B = 1.0   # objective cost (all subsets have unit cost)
USE_LOG_TRICK = True
