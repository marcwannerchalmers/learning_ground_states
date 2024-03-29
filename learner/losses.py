import torch
from torch import nn 

class L1Loss(nn.Module):
    def __init__(self, penalty) -> None:
        super().__init__()
        self.penalty = penalty
        self.mse = nn.MSELoss()

    def forward(self, y_model, y_gt):
        y_pred, weights = y_model
        return self.mse(y_pred, y_gt) + self.penalty*weights.abs().mean(dim=0).sum()

    