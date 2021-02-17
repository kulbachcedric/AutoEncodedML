from datetime import datetime
from time import time

import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from auto_encoder.model import *
from auto_encoder.sklearn import AutoTransformer, ConvolutionalAutoTransformer, SoftmaxClassifier, IdentityTransformer, \
    Transformer
from data.openml import get_openml_data
from data.util import get_train_test_indices
from experiments.util import cv_results_to_df, remove_col_prefix, compute_means
from metrics.reconstruction import ReconstructionError
from metrics.robustness import AdversarialRobustness, NoiseRobustness

np.random.seed(42)


def run_redundancy_test(transformers, clf, scaling='minmax', cv=3, n_informative=2, n_classes=2, n_runs=5,
                        n_samples=1000, score='accuracy', n_clusters_per_class=2, flip_y=0.1, class_sep=1):
    dfs = []
    scaler = MinMaxScaler() if scaling == 'minmax' else StandardScaler
    for transformer in transformers:
        results = {}
        for n_redundant in np.arange(0, 399, 20):
            for run in range(n_runs):
                n_features = n_redundant + n_informative
                x, y = make_classification(n_informative=n_informative, n_redundant=n_redundant, n_features=n_features,
                                           n_samples=n_samples, n_clusters_per_class=n_clusters_per_class,
                                           flip_y=flip_y, class_sep=class_sep, n_classes=n_classes)
                x = scaler.fit_transform(x)
                skf = StratifiedKFold(cv)
                scorer = get_scorer(score)
                for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
                    x_train, x_test = x[train_idx], x[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    x_train_encoded = transformer.fit_transform(x_train)
                    x_test_encoded = transformer.transform(x_test)
                    clf.fit(x_train_encoded, y_train)
                    score_split = scorer(clf, x_test_encoded, y_test)
                    entry_name = f'split{run * cv + fold}_test_{score}'
                    if fold == 0 and run == 0:
                        results[n_redundant] = {entry_name: score_split}
                    else:
                        results[n_redundant][entry_name] = score_split
        df_scenario = pd.DataFrame(results).T
        df_scenario['transformer'] = str(transformer)
        dfs.append(df_scenario)
    df = pd.concat(dfs, axis=0)
    return compute_means(df)


def run_ssl(dataset_ids, transformers, cv=3, labeled_splits=np.arange(0.05, 1.01, 0.05),
            clf=LogisticRegression(penalty='none', max_iter=500), score='accuracy', scaling='minmax'):
    dfs = []
    for dataset_id in dataset_ids:
        for transformer in transformers:
            x, y = get_openml_data(dataset_id, scaling=scaling)
            skf = StratifiedKFold(cv)
            scorer = get_scorer(score)
            results = {}
            for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                x_train_encoded = transformer.fit_transform(x_train)
                x_test_encoded = transformer.transform(x_test)
                for split in labeled_splits:
                    n_samples = round(len(x_train) * split)
                    x_train_split, y_train_split = resample(x_train_encoded, y_train, n_samples=n_samples,
                                                            replace=False,
                                                            stratify=y_train)
                    clf.fit(x_train_split, y_train_split)
                    score_split = scorer(clf, x_test_encoded, y_test)
                    entry_name = f'split{fold}_test_{score}'
                    if fold == 0:
                        results[split] = {entry_name: score_split}
                    else:
                        results[split][entry_name] = score_split

            df_scenario = pd.DataFrame(results).T
            df_scenario['dataset_id'] = dataset_id
            df_scenario['transformer'] = str(transformer)
            dfs.append(df_scenario)
    df = pd.concat(dfs, axis=0)
    return compute_means(df)


def run_clf_test(dataset_ids, scaling='minmax', cv=3, clfs=None, score='accuracy', params=np.arange(0.25, 2.01, 0.25)):
    dfs = []
    for dataset_id in dataset_ids:
        for param in params:
            x, y = get_openml_data(dataset_id, subsample_size=10000, scaling=scaling)
            skf = StratifiedKFold(cv)
            scorer = get_scorer(score)
            results = {}
            transformer = AutoTransformer(hidden_dims=param)
            for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                start = time()
                x_train_encoded = transformer.fit_transform(x_train)
                ae_fit_time = time() - start
                x_test_encoded = transformer.transform(x_test)
                for clf in clfs:
                    results_scenario = {f'split{fold}_ae_fit_time': ae_fit_time}
                    start = time()
                    clf.fit(x_train_encoded, y_train)
                    results_scenario[f'split{fold}_clf_fit_time'] = time() - start
                    start = time()
                    results_scenario[f'split{fold}_test_{score}'] = scorer(clf, x_test_encoded, y_test)
                    results_scenario[f'split{fold}_score_time'] = time() - start

                    if fold == 0:
                        results[str(clf)] = results_scenario
                    else:
                        for key, value in results_scenario.items():
                            results[str(clf)][key] = value

            df_scenario = pd.DataFrame(results).T
            df_scenario['dataset_id'] = dataset_id
            df_scenario['param'] = param
            dfs.append(df_scenario)
    df = pd.concat(dfs, axis=0)
    return compute_means(df)


def run_gridsearch(dataset_ids, scorers, cv=3, reshape=False, scaling='minmax'):
    pipe = Pipeline([
        ('ae', AutoTransformer(max_epochs=100, patience=2, hidden_dims=0.35)),
        ('clf', LogisticRegression(penalty='none', max_iter=500))
    ])
    params = {
        'ae__type': ['ae', 'vae', 'dae', 'sae']
    }
    results = {}
    for dataset_id in dataset_ids:
        print(f'---------Dataset: {dataset_id}---------')
        x, y = get_openml_data(dataset_id, scaling=scaling)
        if reshape:
            x = np.reshape(x, (-1, 28, 28, 1))
        splits = [get_train_test_indices(y)] if cv == 1 else cv
        grid = GridSearchCV(estimator=pipe, param_grid=params, cv=splits,
                            scoring=scorers, refit=False, verbose=2)
        grid.fit(x, y)
        results[dataset_id] = pd.DataFrame(grid.cv_results_)

    df = pd.concat(results)
    df.index.names = ('dataset_id', 'idx')
    df.reset_index(level='idx', drop=True, inplace=True)
    day = datetime.now().strftime('%j')
    df = remove_col_prefix(df)
    df.to_csv(f'gridsearchcv_results_{day}.csv')


def run_training_robustness_test(dataset_ids, transformers, scaling='minmax', cv=3, clfs=None, score='accuracy',
                                 noise_levels=np.arange(0, 1.01, 0.1), corrupt_type='snp'):
    for dataset_id in dataset_ids:
        for noise_level in noise_levels:
            x, y = get_openml_data(dataset_id, scaling=scaling, corrupt_type=corrupt_type, noise_level=noise_level)
            skf = StratifiedKFold(cv)
            scorer = get_scorer(score)
            results = {}


if __name__ == '__main__':
    datasets = [40996, 40668, 1492, 44]
    scorers = {f'accuracy_noise_{s}': NoiseRobustness(scale=s) for s in np.arange(0, 1.01, 0.1)}
    scorer_accuracy = {'accuracy': get_scorer('accuracy')}
    run_gridsearch(datasets, scorers=scorer_accuracy, cv=3, reshape=False)
