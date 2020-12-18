from art.attacks.evasion import ZooAttack, HopSkipJump
from art.estimators.classification import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from data.openml import get_openml_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, get_scorer
from data.util import corrupt_gaussian, corrupt_zero_mask
import numpy as np
from sklearn.pipeline import Pipeline
from auto_encoder.sklearn import AutoTransformer
from sklearn.utils import resample


class AdversarialRobustness:
    def __init__(self, attack='hsj', attack_params=None, sample_size=None):
        self.attack = HopSkipJump if attack == 'hsj' else ZooAttack
        self.attack_params = attack_params
        self.sample_size = sample_size

    def __call__(self, estimator, X, y_true=None, sample_weight=None):
        if self.sample_size:
            X, y_true = resample(X, y_true, n_samples=self.sample_size, replace=False)
        adv_model = SklearnClassifier(estimator)
        attack = self.attack(adv_model, **self.attack_params)
        X_adv = attack.generate(x=X, y=y_true)
        preds_adv = estimator.predict(X_adv)
        fooled_preds = preds_adv != y_true
        adv_perturbation_fooled = (X_adv - X)[fooled_preds]
        return np.mean(np.linalg.norm(adv_perturbation_fooled, axis=-1))


class NoiseRobustness:
    def __init__(self, corruption='zero_mask', scale=0.2, scoring='accuracy'):
        if callable(corruption):
            self.corrupt = corruption
        else:
            self.corrupt = corrupt_zero_mask if corruption == 'zero_mask' else corrupt_gaussian
        if callable(scoring):
            self.score = scoring
        else:
            self.score = get_scorer(scoring)
        self.scale = scale

    def __call__(self, estimator, X, y_true, sample_weight=None):
        X_corrupted = self.corrupt(X, scale=self.scale)
        return self.score(estimator, X_corrupted, y_true)


if __name__ == '__main__':
    x, y = get_openml_data(44)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = Pipeline([('at', AutoTransformer(max_epochs=5)), ('log_reg', LogisticRegression(max_iter=500))])   # LogisticRegression(max_iter=500)
    clf.fit(x_train, y_train)
    adv_metric = AdversarialRobustness(sample_size=10, attack='zoo', attack_params={'nb_parallel': 20, 'binary_search_steps': 20, 'max_iter': 100})
    print(adv_metric(clf, x_test, y_test))
