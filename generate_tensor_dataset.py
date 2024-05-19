import os
import torch
from tensordict import TensorDict
import numpy as np
from util.data import get_correlation, get_couplings, get_data

DATA_PER_SET = 4096

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

def generate_tensor_dataset_new(shape, path_save, prefix):
    data_dict = {"X": [], "Y": []}
    
    folder_seq = "unif_LDS" if prefix == "lds" else "unif_random"

    path = os.path.join("final_data", folder_seq, "data_{}x{}".format(*shape))
    length = 0
    for id in range(1, DATA_PER_SET+1):
        fname = os.path.join(path, "{}_{}x{}_id{}_XX.txt".format('simulation', *shape, id))
        if not os.path.exists(fname):
            continue
        X = get_couplings(path, shape[0], shape[1], id, prefix='simulation')
        Y = get_correlation(path, shape[0], shape[1], id, prefix='simulation')
        if np.isnan(Y).any():
            print("Nan")
            continue
        data_dict["X"].append(X)
        data_dict["Y"].append(Y)
        length += 1

    for key, item in data_dict.items():
        item = np.array(item)
        data_dict[key] = torch.from_numpy(item)

    min_x = torch.min(data_dict["X"], dim=0).values
    max_x = torch.max(data_dict["X"], dim=0).values
    min_y = torch.min(data_dict["Y"])
    max_y = torch.max(data_dict["Y"])
    data_dict["extrema_X"] = torch.stack([min_x, max_x])
    data_dict["extrema_Y"] = torch.tensor([min_y, max_y])
    data_dict["N_data"] = length
    data = TensorDict(data_dict, batch_size=[])
    torch.save(data, os.path.join(path_save, "{}_{}x{}.td".format(prefix, *shape)))


def main():
    rows = (4, 9)
    shadow_sizes = [50, 100, 250, 500, 1000]
    for i in range(rows[0], rows[1]+1):
        #generate_tensor_dataset((i,5), path_save="data_torch", shadow_sizes=shadow_sizes)
        generate_tensor_dataset_new((i,5), path_save="data_torch", prefix='lds')
        generate_tensor_dataset_new((i,5), path_save="data_torch", prefix='rand')

if __name__ == "__main__":
    main()