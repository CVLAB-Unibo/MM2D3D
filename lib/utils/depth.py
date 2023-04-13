import torch
from torch import nn


class FilterDepth(nn.Module):
    def __init__(self, kernel: int = 9, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold
        self.max_pool = nn.MaxPool2d(kernel, 1, kernel // 2)

    def forward(self, x):

        orig_depth = x
        max_depth = x.max()

        # compute min pooling
        x = self.max_pool(
            torch.where(
                x > 0, max_depth - x, torch.tensor(0.0, dtype=x.dtype, device=x.device)
            )
        )
        pooled_depth = torch.where(
            x > 0, max_depth - x, torch.tensor(0.0, dtype=x.dtype, device=x.device)
        )

        # apply threshold
        mask = orig_depth > 0
        diff = torch.abs(pooled_depth - orig_depth)[mask] / pooled_depth[mask]
        filtered_depth = torch.zeros_like(orig_depth)
        filtered_depth[mask] = torch.where(
            diff < self.threshold,
            orig_depth[mask],
            torch.tensor(0.0, dtype=orig_depth.dtype, device=orig_depth.device),
        )

        return filtered_depth
