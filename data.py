import torch
import os
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, shape) -> None:
        super().__init__()
        self.path = "new_data" + os.sep + "data_"+str(shape[0])+"x"+str(shape)

def main():
    pass

if __name__ == "__main__":
    main()