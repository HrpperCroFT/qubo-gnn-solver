import torch

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32

NUMBER_EPOCHS = int(4e5)
LEARNING_RATE = 1e-5
PROB_THRESHOLD = 0.5
TOL = 1e-4
PATIENCE = 100
DIM_EMBEDDING = 10
HIDDEN_DIM = 32
DROPOUT = 0.5
NUMBER_CLASSES = 1
PENALTY_COEFF = 2.5
LOG_EVERY = 1000
