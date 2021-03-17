import numpy as np
from art.attacks.evasion import ZooAttack, HopSkipJump
from art.estimators.classification import SklearnClassifier
from sklearn.metrics import get_scorer
from sklearn.utils import resample

from data.util import corrupt_gaussian, corrupt_snp


def get_adversarial_examples(estimator, X, y_true=None, attack='hsj', **attack_params):

    adv_model = SklearnClassifier(estimator, clip_values=(0, 1))

    if attack == 'zoo':
        zoo_params_default = {'nb_parallel': int(0.3 * np.prod(X.shape[1:])),
                              'binary_search_steps': 10,
                              'max_iter': 100}
        for key, value in zoo_params_default.items():
            if key not in attack_params:
                attack_params[key] = value
        attack_gen = ZooAttack(adv_model, **attack_params)
    else:
        attack_gen = HopSkipJump(adv_model, **attack_params)

    try:
        X_adv = attack_gen.generate(x=X, y=y_true)
        X_adv = np.nan_to_num(X_adv)
        preds_adv = estimator.predict(X_adv)
        return X_adv, preds_adv
    except:
        print(f'adversarial attack on {str(estimator)} failed')


class AdversarialRobustness:
    def __init__(self, attack='hsj', attack_params=None, sample_size=100):
        self.attack = attack
        self.attack_params = attack_params if attack_params else {}
        self.sample_size = sample_size

    def __call__(self, estimator, X, y_true=None, sample_weight=None):
        if self.sample_size:
            X, y_true = resample(X, y_true, n_samples=self.sample_size, replace=False, stratify=y_true)
            X_adv, preds_adv = get_adversarial_examples(estimator, X, y_true, attack=self.attack, **self.attack_params)
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
