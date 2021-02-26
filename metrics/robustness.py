import numpy as np
from art.attacks.evasion import ZooAttack, HopSkipJump
from art.estimators.classification import SklearnClassifier
from sklearn.metrics import get_scorer
from sklearn.utils import resample

from data.util import corrupt_gaussian, corrupt_snp


class AdversarialRobustness:
    def __init__(self, attack='hsj', attack_params=None, sample_size=None, save_examples=False):
        self.attack = attack
        self.attack_params = attack_params if attack_params else {}
        self.sample_size = sample_size
        self.save_examples = save_examples
        self.i = 0

    def __call__(self, estimator, X, y_true=None, sample_weight=None):
        if self.sample_size:
            X, y_true = resample(X, y_true, n_samples=self.sample_size, replace=False, stratify=y_true)
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

        try:
            X_adv = attack.generate(x=X, y=y_true)
            X_adv = np.nan_to_num(X_adv)
            preds_adv = estimator.predict(X_adv)
            if self.save_examples:
                n_features = np.prod(X.shape[1:])
                transformer_name = get_type(str(estimator.steps[0][1]))
                results = {'true_label': y_true, 'predicted_label': preds_adv, 'adv_examples': X_adv}
                np.save(f'adv_data/{n_features}_{transformer_name}_{self.i}.npy', results, allow_pickle=True)
                self.i += 1
            fooled_preds = preds_adv != y_true
            adv_perturbation_fooled = (X_adv - X)[fooled_preds]
            return np.average(np.linalg.norm(adv_perturbation_fooled, axis=-1), weights=sample_weight)
        except:
            print('Adversarial attack failed. Setting robustness to 1.')
            return 1


def get_type(s):
    if 'vae' in s:
        return 'VAE'
    elif 'sae' in s:
        return 'SAE'
    elif 'dae' in s:
        return 'DAE'
    elif 'Identity' in s:
        return 'None'
    else:
        return 'AE'


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
