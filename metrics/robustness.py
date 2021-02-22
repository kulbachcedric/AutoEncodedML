import numpy as np
from art.attacks.evasion import ZooAttack, HopSkipJump
from art.estimators.classification import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from auto_encoder.sklearn import AutoTransformer
from data.openml import get_openml_data
from data.util import corrupt_gaussian, corrupt_snp


class AdversarialRobustness:
    def __init__(self, attack='hsj', attack_params=None, sample_size=None):
        self.attack = attack
        self.attack_params = attack_params if attack_params else {}
        self.sample_size = sample_size

    def __call__(self, estimator, X, y_true=None, sample_weight=None):
        if self.sample_size:
            X, y_true = resample(X, y_true, n_samples=self.sample_size, replace=False)
        adv_model = SklearnClassifier(estimator)
        if self.attack == 'zoo':
            zoo_params_default = {'nb_parallel': int(0.3 * np.prod(X.shape[1:])),
                                  'binary_search_steps': 10,
                                  'max_iter': 100}
            for key, value in zoo_params_default.items():
                if key not in self.attack_params:
                    self.attack_params[key] = value
            attack = ZooAttack(adv_model, **self.attack_params)
        else:
            attack = HopSkipJump(adv_model, **self.attack_params)

        X_adv = attack.generate(x=X, y=y_true)
        preds_adv = estimator.predict(X_adv)
        fooled_preds = preds_adv != y_true
        adv_perturbation_fooled = (X_adv - X)[fooled_preds]
        return np.average(np.linalg.norm(adv_perturbation_fooled, axis=-1), weights=sample_weight)


class NoiseRobustness:
    def __init__(self, corruption='zero_mask', scale=0.2, scoring='accuracy'):
        if callable(corruption):
            self.corrupt = corruption
        else:
            self.corrupt = corrupt_snp if corruption == 'zero_mask' else corrupt_gaussian
        if callable(scoring):
            self.scorer = scoring
        else:
            self.scorer = get_scorer(scoring)
        self.scale = scale

    def __call__(self, estimator, X, y_true, sample_weight=None):
        X_corrupted = self.corrupt(X, scale=self.scale)
        return self.scorer(estimator, X_corrupted, y_true)
