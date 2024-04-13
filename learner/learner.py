from pathlib import Path
import sys
import fastai
import torch
from torch import nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from fastai.vision.all import *
from functools import partial
from model.model import SimpleFullDNN, SeparateFullDNN

from data import get_train_test_set
from learner.losses import L1Loss, BTLoss, RMSE, Metric
from util.helper import update_cfg

# torch.multiprocessing.set_sharing_strategy("file_system")

def init_learner(cfg: OmegaConf, test: bool = False) -> fastai.learner.Learner:
    """Initialize a fastai learner object for the multi multiplet model. The learner object is used to train the model. The function returns the learner object. The function also saves the learner object in the log folder.
    Args:
        cfg: configuration object
        test: if True, the learner object is initialized for testing
    Returns: fastai learner object
    """
    root_path = Path(__file__).parents[2]

    model = SeparateFullDNN(**cfg.model_parameters)

    cfg = update_cfg(cfg, model.gm)

    # get training/test data
    train_loader, test_loader = get_train_test_set(**cfg.ds_parameters)

    dls = DataLoaders(train_loader, test_loader)
    
    if cfg.learner_parameters.init_xavier:
        model.init_xavier()

    opt_func = partial(
        OptimWrapper,
        opt=torch.optim.AdamW,
        **cfg.optim_parameters
    )
    
    loss_fn = L1Loss(cfg.learner_parameters.l1_penalty)
    callbacks = [
        SaveModelCallback(fname=cfg.lognames.best_model_file, with_opt=True, reset_on_fit=False),
        ReduceLROnPlateau(patience=cfg.learner_parameters.lr_reduce_epochs, factor=2, reset_on_fit=False),
    ]

    mse = nn.MSELoss(reduction='mean')

    rmse = RMSE(reduction='mean')
    
    bt_mse = BTLoss(cfg.ds_parameters.norm_y, mse)
    bt_l1 = BTLoss(cfg.ds_parameters.norm_y, loss_fn)

    metrics = [Metric(rmse), loss_fn, Metric(bt_mse), bt_l1]

    learner = Learner(dls,
                      model,
                      opt_func=opt_func,
                      loss_func=loss_fn,
                      metrics=metrics,
                      #cbs=callbacks,
                      path=root_path,
                      model_dir=cfg.path_model)
    return learner
