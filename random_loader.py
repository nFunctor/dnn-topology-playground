from torch.utils.data import TensorDataset, DataLoader
from torch import rand


class RandomLoader:
    def __init__(self, dim: int, row_count: int):
        self.dim = dim
        self.data = DataLoader(
            TensorDataset(rand(row_count, dim))
        )
