from torch.utils.data import TensorDataset, DataLoader
import torch

def get_random_loader(row_count: int, dim: int) -> DataLoader:
    random_points = (1-2*torch.rand(row_count * dim)).view((row_count, dim))
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

#if you ever want to test
#print(get_random_loader(3,4).dataset[1])