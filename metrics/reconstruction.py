from sklearn.base import TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline


def find_transformer(pipe):
    for step in pipe.named_steps.values():
        if isinstance(step, TransformerMixin):
            return step


class ReconstructionError:
    scorers = {'mse': mean_squared_error,
               'mae': mean_absolute_error}

    def __init__(self, score='mse'):
        self.scorer = ReconstructionError.scorers[score]

    def __call__(self, estimator, X, y_true=None, sample_weight=None):
        transformer = find_transformer(estimator) if isinstance(estimator, Pipeline) else estimator
        X_transformed = transformer.transform(X)
        X_reconstructed = transformer.inverse_transform(X_transformed)
        return self.scorer(y_true=X, y_pred=X_reconstructed, sample_weight=sample_weight)

