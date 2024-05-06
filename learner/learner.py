from pathlib import Path
import sys
import fastai
import torch
from torch import nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from fastai.vision.all import *
from functools import partial
from model.model import SimpleFullDNN, CombinedFullDNN

from data import get_train_test_set
from learner.losses import L1Loss, BTLoss, RMSE, Metric
from util.helper import update_cfg
from learner.callbacks import FastAIPruningCallback

# torch.multiprocessing.set_sharing_strategy("file_system")

def init_learner(cfg: OmegaConf, test: bool = False, trial=None) -> fastai.learner.Learner:
    """Initialize a fastai learner object for the multi multiplet model. The learner object is used to train the model. The function returns the learner object. The function also saves the learner object in the log folder.
    Args:
        cfg: configuration object
        test: if True, the learner object is initialized for testing
        trial: For hyperparameter tuning
    Returns: fastai learner object
    """
    root_path = Path(__file__).parents[2]

    # construct model and update geometry parameters
    model = CombinedFullDNN(**cfg.model_parameters)
    cfg = update_cfg(cfg, model.gm)

    # get training/test data
    # note: can probably do more elegantly later w. train, validate, test
    mode = 'test' if test else 'train'
    train_loader, validation_loader, _ = get_train_test_set(**cfg.ds_parameters, mode=mode)

    dls = DataLoaders(train_loader, validation_loader)
    
    # initialize model
    if cfg.learner_parameters.init_xavier:
        model.init_xavier()

    # initialize optimizer
    opt_func = partial(
        OptimWrapper,
        opt=torch.optim.AdamW,
        **cfg.optim_parameters
    )
    
    # train on l1 loss + training objective
    loss_fn = L1Loss(cfg.learner_parameters.l1_penalty)
    callbacks = [
        #SaveModelCallback(fname=cfg.lognames.best_model_file, with_opt=True, reset_on_fit=False),
    ]

    # to get better convergence close to the optimum
    if not test:
        callbacks.append(ReduceLROnPlateau(patience=cfg.learner_parameters.lr_reduce_epochs, factor=2, reset_on_fit=False))

    # for hyperparameter tuning (currently not used)
    if trial is not None:
        callbacks.append(partial(FastAIPruningCallback, trial=trial, monitor='exp_rmspe'))

    mse = nn.MSELoss(reduction='mean')

    rmse = RMSE(reduction='mean')
    
    bt_mse = BTLoss(cfg.ds_parameters.norm_y, mse)
    bt_l1 = BTLoss(cfg.ds_parameters.norm_y, loss_fn)

    # define metrics which are shown during training for validation set
    metrics = [Metric(rmse), loss_fn, Metric(bt_mse), bt_l1]

    learner = Learner(dls,
                      model,
                      opt_func=opt_func,
                      loss_func=loss_fn,
                      metrics=metrics,
                      cbs=callbacks,
                      path=root_path,
                      model_dir=cfg.path_model)
    return learner
