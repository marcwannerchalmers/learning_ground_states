from fastai.vision.all import *
#import optuna

class FastAIPruningCallback(TrackerCallback):
    def __init__(self, learn, trial, monitor):
        # type: (Learner, optuna.trial.Trial, str) -> None

        super(FastAIPruningCallback, self).__init__(learn, monitor)

        self.trial = trial

    def on_epoch_end(self, epoch, **kwargs):
        # type: (int, Any) -> None

        value = self.get_monitor_value()
        if value is None:
            return

        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            message = 'Trial was pruned at epoch {}.'.format(epoch)
            raise optuna.structs.TrialPruned(message)