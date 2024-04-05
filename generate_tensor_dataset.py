import os
import torch
from tensordict import TensorDict
import numpy as np
from util.data import get_energy, get_couplings

DATA_PER_SET = 500

def generate_tensor_dataset(shape, path_save):
    path_load = "new_data" + os.sep + "data_"+str(shape[0])+"x"+str(shape[1])
    X = []
    Y = []
    for id in range(1,DATA_PER_SET+1):
        x = get_couplings(path_load, row=shape[0], col=shape[1], id_=id, prefix='simulation')
        y = get_energy(path_load, row=shape[0], col=shape[1], id_=id, prefix='simulation')
        X.append(x)
        Y.append(y)

    X = torch.tensor(X)
    Y = torch.from_numpy(np.array(Y))
    min_x = torch.min(X, dim=0).values
    max_x = torch.max(X, dim=0).values
    min_y = torch.min(Y)
    max_y = torch.max(Y)
    data = TensorDict({"X": X, "Y": Y, 
                       "extrema_X": torch.stack([min_x, max_x]),
                       "extrema_Y": torch.tensor([min_y, max_y])}, batch_size=[])
    torch.save(data, os.path.join(path_save, "{}x{}.td".format(*shape)))


def main():
    rows = (5, 9)
    for i in range(rows[0], rows[1]+1):
        generate_tensor_dataset((i,5), path_save="data_torch")

if __name__ == "__main__":
    main()