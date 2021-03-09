from torch.utils.data import TensorDataset, DataLoader
import torch

def get_random_loader(row_count: int, dim: int) -> DataLoader:
    random_points = torch.rand(row_count * dim).view((row_count, dim))
    return DataLoader(
        TensorDataset(
            random_points,
            torch.where(
                random_points.pow(2).sum(axis=1) < 1,
                torch.ones(row_count, dtype=torch.long),
                torch.zeros(row_count, dtype=torch.long)
            )
        )
    )
