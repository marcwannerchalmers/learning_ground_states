from fastai.vision.all import *
import optuna
from learner.learner import init_learner
import copy
from omegaconf import OmegaConf
import hydra

def get_objective(cfg):
    local_params = cfg.model_parameters.local_parameters
    def objective(trial: optuna.Trial):
        width = trial.suggest_int('width', *local_params.width, log=True)
        depth = trial.suggest_int('depth', *local_params.depth)
        l2_penalty = trial.suggest_float('l2_penalty', *cfg.optim_parameters.weight_decay)
        l1_penalty = trial.suggest_float('l1_penalty', *cfg.learner_parameters.l1_penalty)

        cfg.model_parameters.local_parameters.width = width
        cfg.model_parameters.local_parameters.depth = depth
        cfg.optim_parameters.weight_decay = l2_penalty
        cfg.learner_parameters.l1_penalty = l1_penalty

        learner: Learner = init_learner(cfg)
        learner.fit(cfg.learner_parameters.n_epochs_max)
        return learner.validate()[-1].item()
    
    return objective
    
def optimize_hyperparameters(cfg):
    objective = get_objective(cfg)
    study = optuna.create_study(objective)
    return study.best_trial

@hydra.main(version_base=None, config_path="conf", config_name="config_hyper.yaml")
def main(cfg : OmegaConf) -> None:
    res = optimize_hyperparameters
    print(res)


# test
if __name__ == "__main__":
    main()
