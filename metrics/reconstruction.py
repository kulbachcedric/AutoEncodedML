from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from tensorflow.keras.losses import binary_crossentropy
import numpy as np

from auto_encoder.sklearn import Transformer


def find_transformer(pipe):
    for step in pipe.named_steps.values():
        if isinstance(step, Transformer):
            return step


def log_loss(y_true, y_pred, sample_weight=None):
    bce = binary_crossentropy(y_true=y_true, y_pred=y_pred).numpy()
    if sample_weight:
        bce = bce.dot(sample_weight)
    return np.mean(bce)


class ReconstructionError:
    scorers = {'mse': mean_squared_error,
               'mae': mean_absolute_error,
               'bce': log_loss}

    def __init__(self, score='bce'):
        self.scorer = ReconstructionError.scorers[score]

    def __call__(self, estimator, X, y_true=None, sample_weight=None):
        transformer = find_transformer(estimator) if isinstance(estimator, Pipeline) else estimator
        X_transformed = transformer.transform(X)
        X_reconstructed = transformer.inverse_transform(X_transformed)
        return self.scorer(y_true=X, y_pred=X_reconstructed, sample_weight=sample_weight)

