import torch
from torch import nn 
from abc import abstractmethod

class Transform(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def inverse(self, x):
        pass

class Identity(Transform):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        return x
    
    def inverse(self, x):
        return x
    
class UnitInterval(Transform):
    def __init__(self, extrema, **kwargs) -> None:
        super().__init__()
        self.extrema = extrema

    def forward(self, x):
        min, max = self.extrema
        return (x - min)/(max-min) 
    
    def inverse(self, x):
        min, max = self.extrema
        return x * (max - min) + min

# test
def main():
    tf = Identity()
    print(tf(torch.Tensor([1])))

if __name__ == "__main__":
    main()