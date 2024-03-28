import torch
from torch import nn 
from abc import abstractmethod

class Transform(nn.Module):
    def __init__(self, extrema) -> None:
        self.extrema = extrema
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def inverse(self, x):
        pass

class Identity(Transform):
    def __init__(self, extrema, **kwargs) -> None:
        super().__init__(extrema)

    def forward(self, x):
        return x
    
    def inverse(self, x):
        return x

# test
def main():
    tf = Identity()
    print(tf(torch.Tensor([1])))

if __name__ == "__main__":
    main()