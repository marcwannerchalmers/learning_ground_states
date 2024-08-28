from fastai.vision.all import *
import os
import hydra
from omegaconf import OmegaConf

from learner.learner import init_learner
import copy
from data import get_train_test_set
from model.geometry import GridMap
from model.model import CombinedFullDNN

# stores results in same format as lllewis
def save_results(path, errors, gm: GridMap, split, seq):
    filename = "dl_results_{}_{}x{}_{}_{}_data.txt".format(*gm.shape, seq, split)
    file_save = os.path.join(path, filename)
    with open(file_save, "w") as f:
        for error, edge in zip(errors, gm.edges):
            q1, q2 = edge
            print("(q1, q2) = ({}, {})".format(q1, q2), file=f)
            print("({}, {})".format(0, error), file=f)

def evaluate_all(cfg : OmegaConf) -> None:
    rows = (4, 9)
    train_splits = [0.1, 0.3, 0.5, 0.7, 0.9]
    for row in range(rows[0], rows[1]+1):
        cfg.ds_parameters.shape = [row, 5]
        cfg.ds_parameters.y_indices = "edges"
        cfg.model_parameters.geometry_parameters.shape = [row, 5]
        for split in train_splits:
            cfg.ds_parameters.split = split
            errors, gm = evaluate(cfg)
            save_results(cfg.path_eval, errors, gm, split, cfg.ds_parameters.seq)

# stores results in same format as lllewis
def save_results_new(cfg, errors_train, errors_test, gm: GridMap):
    shape = cfg.ds_parameters.shape
    path = cfg.path_eval
    split = cfg.ds_parameters.split
    delta1 = cfg.model_parameters.geometry_parameters.delta1
    seq = cfg.ds_parameters.seq
    filename = "dl_results_{}_{}x{}_split_{}_delta1_{}.txt".format(seq, *shape, split, delta1)
    file_save = os.path.join(path, filename)
    with open(file_save, "w") as f:
        for error_train, error_test, edge in zip(errors_train, errors_test, gm.edges):
            q1, q2 = edge
            print("(q1, q2) = ({}, {})".format(q1, q2), file=f)
            print("({}, {})".format(error_train, error_test), file=f)

# compute rmses          
def evaluate(cfg):

    device = torch.device(cfg.ds_parameters.device)
    learner: Learner = init_learner(cfg, test=True)
    with learner.no_bar():
        learner.fit(cfg.learner_parameters.n_epochs_max)
    model: CombinedFullDNN = learner.model.eval()
    train_loader, _, test_loader = get_train_test_set(**cfg.ds_parameters, mode='test')
    errors_test = torch.zeros((model.n_terms,)).to(device)
    errors_train = torch.zeros((model.n_terms,)).to(device)

    with torch.no_grad():
        # here I compute the average prediction error
        for data in test_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            errors_test += ((model(x)[0] - y)**2).mean(dim=0) # mean is computed over batches (i.e. divide by batch size)

    errors_test = torch.sqrt(errors_test / len(test_loader)) # mean is computed over number of batches (len(loader)) --> divide by N.o. batches
    
    with torch.no_grad():
        # here I compute the average prediction error
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            errors_train += ((model(x)[0] - y)**2).mean(dim=0) # mean is computed over batches (i.e. divide by batch size)

    errors_train = torch.sqrt(errors_train / len(test_loader)) # mean is computed over number of batches (len(loader)) --> divide by N.o. batches
    
    return errors_train, errors_test, model.gm


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg : OmegaConf) -> None:
    print("Seq: ", cfg.ds_parameters.seq)
    print("Shape: {}x{}".format(*cfg.ds_parameters.shape))
    print("Delta1: ", cfg.model_parameters.geometry_parameters.delta1)
    print("Split: ", cfg.ds_parameters.split)
    errors_train, errors_test, gm = evaluate(cfg)
    save_results_new(cfg, errors_train, errors_test, gm)
            
    
if __name__ == "__main__":
    main()