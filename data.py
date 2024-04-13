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
                 y_indices,
                 shadow_size=0,
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
        y_key = "Y" if shadow_size == 0 else "Y{}".format(shadow_size)

        assert y_key in self.data.keys(), "Requested Shadow size not in data"
        assert y_indices is not None, "State which indices the pauli terms correspond to"

        self.Y = self.norm_y(self.data[y_key][start:end]).float().to(device)

        # turn Y into list of w.r.t. parameters given
        self.Y = torch.stack([self.Y[...,ind[0], ind[1]] for ind in y_indices],
                           dim=-1)

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
def get_train_test_set(path, 
                       shape, 
                       n_data,
                       split=0.5, 
                       batch_size=16,
                       y_indices=None,
                       shadow_size=0,
                       norm_x="id", 
                       norm_y="id", 
                       device="cpu",
                       tf_args_x={}, 
                       tf_args_y={}):
    n_train = int(n_data * split)
    train_set = Data(path, 
                     shape, 
                     0, 
                     n_train,
                     y_indices,
                     shadow_size, 
                     norm_x, 
                     norm_y, 
                     device, 
                     tf_args_x, 
                     tf_args_y)
    
    test_set = Data(path, 
                    shape, 
                    n_train, 
                    n_data, 
                    y_indices,
                    0, # always exact data for test
                    norm_x, 
                    norm_y, 
                    device, 
                    tf_args_x, 
                    tf_args_y)

    return  DataLoader(train_set, 
                      batch_size=batch_size, 
                      shuffle=True), \
            DataLoader(test_set, 
                       batch_size=batch_size)

def main():
    pass

if __name__ == "__main__":
    main()