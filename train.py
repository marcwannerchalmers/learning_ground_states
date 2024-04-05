import hydra
from omegaconf import DictConfig, OmegaConf
from learner.learner import init_learner
import torch
import os


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def train(cfg : OmegaConf) -> None:
    learner = init_learner(cfg)
    lr = cfg.optim_parameters.lr

    lr = learner.lr_find()
    print(f"Found learning rate is: {lr}")
    learner.fit(cfg.learner_parameters.n_epochs_max)


if __name__ == "__main__":
    train()