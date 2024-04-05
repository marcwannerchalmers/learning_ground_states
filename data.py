import torch
import os
from torch.utils.data import Dataset, DataLoader
from util.helper import get_transform

class Data(Dataset):
    def __init__(self, 
                 path, 
                 shape, 
                 start, 
                 end, 
                 norm_x="id", 
                 norm_y="id", 
                 device="cpu",
                 tf_args_x={}, 
                 tf_args_y={}) -> None:
        
        super().__init__()
        device = torch.device(device)
        self.data = torch.load(os.path.join(path, "{}x{}.td".format(*shape)))
        self.norm_x = get_transform(norm_x, extrema=self.data["extrema_X"], **tf_args_x).float().to(device)
        self.norm_y = get_transform(norm_x, extrema=self.data["extrema_Y"], **tf_args_y).float().to(device)
        self.X = self.norm_x(self.data["X"][start:end]).float().to(device)
        self.Y = self.norm_y(self.data["Y"][start:end]).float().to(device)

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
def get_train_test_set(path, 
                       shape, 
                       n_data,
                       split=0.5, 
                       batch_size=16,
                       norm_x="id", 
                       norm_y="id", 
                       device="cpu",
                       tf_args_x={}, 
                       tf_args_y={}):
    n_train = int(n_data * split)
    train_set = Data(path, shape, 0, n_train, norm_x, norm_y, device, tf_args_x, tf_args_y)
    test_set = Data(path, shape, n_train, n_data, norm_x, norm_y, device, tf_args_x, tf_args_y)

    return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size)

def main():
    pass

if __name__ == "__main__":
    main()