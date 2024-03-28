import torch
import os
from torch.utils.data import Dataset, DataLoader
from util.transforms import Identity

class Data(Dataset):
    def __init__(self, 
                 path, 
                 shape, 
                 start, 
                 end, 
                 norm_x=Identity, 
                 norm_y=Identity, 
                 tf_args_x={}, 
                 tf_args_y={}) -> None:
        
        super().__init__()
        self.data = torch.load(os.path.join(path, "{}x{}.td".format(*shape)))
        self.norm_x = norm_x(self.data["extrema_X"], **tf_args_x)
        self.norm_y = norm_y(self.data["extrema_Y"], **tf_args_y)
        self.X = norm_x(self.data["X"][start:end])
        self.Y = norm_y(self.data["Y"][start:end])

    def __len__(self):
        return len(self.data["Y"])
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
def get_train_test_set(path, 
                       shape, 
                       n_data,
                       split=0.5, 
                       batch_size=16,
                       norm_x=Identity, 
                       norm_y=Identity, 
                       tf_args_x={}, 
                       tf_args_y={}):
    n_train = int(n_data * split)
    train_set = Data(path, shape, 0, n_train, norm_x, norm_y, tf_args_x, tf_args_y)
    test_set = Data(path, shape, n_train, n_data, norm_x, norm_y, tf_args_x, tf_args_y)

    return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size)

def main():
    pass

if __name__ == "__main__":
    main()