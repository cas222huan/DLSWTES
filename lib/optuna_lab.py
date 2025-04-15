import optuna
import numpy as np

# self defined early stop for optuna
class early_stop_callback:
    def __init__(self, threshold=10, direction='min'):
        assert threshold > 0, "Param <threshold> must be positive"
        assert direction in ['min', 'max'], "Param <direction> must be 'min' or 'max'"

        self.threshold = threshold
        self.best_score = np.Inf
        self.counts = 0
        self.multiplier = 1 if direction=='min' else -1

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        value = trial.value * self.multiplier
        if value < self.best_score:
            self.best_score = value
            self.counts = 0
        else:
            self.counts += 1
        if self.counts >= self.threshold:
            study.stop()
            print(f"Early stop at {trial.number}th trial")