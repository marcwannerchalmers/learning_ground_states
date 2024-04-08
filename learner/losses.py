import torch
from torch import nn 
from util.transforms import Transform
from util.helper import get_transform

class L1Loss(nn.Module):
    def __init__(self, penalty) -> None:
        super().__init__()
        self.penalty = penalty
        self.mse = nn.MSELoss()

    def forward(self, y_model, y_gt):
        y_pred, weights = y_model
        return self.mse(y_pred, y_gt) + self.penalty*weights.abs().mean(dim=0).sum()
    
class BTLoss(nn.Module):
    def __init__(self, tf: str, loss_fn, **tf_args) -> None:
        super().__init__()
        self.tf = get_transform(tf, **tf_args)
        self.loss_fn = loss_fn

    def forward(self, y_pred, y):
        y = self.tf.inverse(y)
        y_pred = self.tf.inverse(y_pred)
        return self.loss_fn(y_pred, y)
    