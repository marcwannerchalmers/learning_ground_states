import torch
import os
from torch.utils.data import Dataset, DataLoader
from util.helper import get_transform
import copy

# dataset object
# shadow_size=0 <--> exact correlations
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

# returns dataloader objects for training, validation and test set
def get_train_test_set(path, 
                       shape, 
                       n_data,
                       split=0.5, 
                       validation_split=0.8,
                       batch_size=16,
                       y_indices=None,
                       shadow_size=0,
                       norm_x="id", 
                       norm_y="id", 
                       device="cpu",
                       tf_args_x={}, 
                       tf_args_y={},
                       mode='train'):
    
    # only use training data for train/validation set
    n_data_train = int(n_data * split)

    if mode == 'test':
        validation_split = 1

    n_train = int(n_data_train * validation_split)
    bs_validate = n_data_train-n_train 

    if mode == 'test':
        bs_validate = batch_size # error if is set to 0

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
    
    validation_set = Data(path, 
                     shape, 
                     n_train, 
                     n_data_train,
                     y_indices,
                     shadow_size, 
                     norm_x, 
                     norm_y, 
                     device, 
                     tf_args_x, 
                     tf_args_y)
    
    test_set = Data(path, 
                    shape, 
                    n_data_train, 
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
            DataLoader(validation_set, 
                      batch_size=bs_validate), \
            DataLoader(test_set, 
                       batch_size=batch_size)

def main():
    pass

if __name__ == "__main__":
    main()