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
        mse = self.mse(y_pred, y_gt)
        return mse + self.penalty * weights.mean()
    
class BTLoss(nn.Module):
    def __init__(self, tf: str, loss_fn, **tf_args) -> None:
        super().__init__()
        self.tf = get_transform(tf, **tf_args)
        self.loss_fn = loss_fn

    def forward(self, y_pred, y):
        y = self.tf.inverse(y)
        y_pred = self.tf.inverse(y_pred)
        return self.loss_fn(y_pred, y)
    
class RMSE(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((input-target)**2).mean(dim=0).sqrt().mean()
        #return torch.sqrt(super().forward(input, target))

# loss that only takes the first parameter of the input tuple
class Metric(nn.Module):
    def __init__(self, loss_fn, ind=0) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.ind = ind

    def forward(self, input, target):
        return self.loss_fn(input[self.ind], target)

    