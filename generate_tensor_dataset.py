import os
import torch
from tensordict import TensorDict
import numpy as np
from util.data import get_energy, get_couplings, get_data

DATA_PER_SET = 500

# store the old data in more convenient format

def generate_tensor_dataset(shape, path_save, shadow_sizes=[500]):
    data_dict = {}
    for size in shadow_sizes:
        s_id = "Y{}".format(size)
        data_dict["X"], data_dict[s_id], data_dict["Y"] = get_data(nrow=shape[0], 
                                                                ncol=shape[1], 
                                                                shadow_size=size,
                                                                data_name='new',
                                                                normalize=False)
    for key, item in data_dict.items():
        data_dict[key] = torch.from_numpy(item)
    min_x = torch.min(data_dict["X"], dim=0).values
    max_x = torch.max(data_dict["X"], dim=0).values
    min_y = torch.min(data_dict["Y"])
    max_y = torch.max(data_dict["Y"])
    data_dict["extrema_X"] = torch.stack([min_x, max_x])
    data_dict["extrema_Y"] = torch.tensor([min_y, max_y])
    data = TensorDict(data_dict, batch_size=[])
    torch.save(data, os.path.join(path_save, "{}x{}.td".format(*shape)))


def main():
    rows = (4, 9)
    shadow_sizes = [50, 100, 250, 500, 1000]
    for i in range(rows[0], rows[1]+1):
        generate_tensor_dataset((i,5), path_save="data_torch", shadow_sizes=shadow_sizes)

if __name__ == "__main__":
    main()