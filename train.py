import hydra
from omegaconf import DictConfig, OmegaConf
from learner.learner import init_learner
import torch


@hydra.main(config_path="conf", config_name="config.yaml")
def train(cfg : OmegaConf) -> None:
    print(cfg.model_parameters.local_parameters.act_fun()(torch.tensor(10)))
    learner = init_learner(cfg)
    lr = cfg.optim_parameters.lr
    if lr is None:
        lr = learner.lr_find()
    print(f"Found learning rate is: {lr}")
    learner.fit(cfg.learner_parameters.n_epochs_max)


if __name__ == "__main__":
    train()