import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import reduce
import os

# not generic, but does the trick for now
def generate_config(cfg_orig: dict, path_save, seq, shape, delta1, split):

    fname = "config_{}_{}x{}_delta1_{}_split_{}.yaml".format(seq, *shape, delta1, split)
    cfg_orig["ds_parameters"]["seq"] = seq
    cfg_orig["ds_parameters"]["shape"] = [shape[0], shape[1]]
    cfg_orig["model_parameters"]["geometry_parameters"]["shape"] = [shape[0], shape[1]]
    cfg_orig["model_parameters"]["geometry_parameters"]["delta1"] = delta1
    cfg_orig["ds_parameters"]["split"] = split
    cfg_orig["path_eval"] = "results/dl_{}/res_{}x{}/".format(seq, *shape)

    file = os.path.join(path_save, fname)
    with open(file, 'w') as outfile:
        yaml.dump(cfg_orig, outfile, default_flow_style=False)

@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg : OmegaConf):
    base_config = OmegaConf.to_container(cfg)
    seqs = ["lds", "rand"]
    shapes = [[i, 5] for i in range(4, 10)]
    delta1s = list(range(6))
    splits = [0.1, 0.3, 0.5, 0.7, 0.9]
    for seq in seqs:
        for shape in shapes:
            for delta1 in delta1s:
                for split in splits:
                    arch = "T4" if shape[0] < 8 else "A40"
                    generate_config(base_config, "dl_{}_{}/".format(seq, arch), seq, shape, delta1, split)

if __name__ == "__main__":
    main()