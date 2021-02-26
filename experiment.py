from datetime import datetime
from time import time

import pandas as pd
from pipelinehelper import PipelineHelper
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.manifold import Isomap
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tpot import TPOTClassifier
from xgboost.sklearn import XGBClassifier

from auto_encoder.model import *
from auto_encoder.sklearn import AutoTransformer, ConvolutionalAutoTransformer, SoftmaxClassifier, IdentityTransformer, \
    Transformer
from data.openml import get_openml_data
from data.util import get_train_test_indices, corrupt_snp, corrupt_gaussian
from experiments.util import cv_results_to_df, remove_col_prefix, compute_means
from metrics.reconstruction import ReconstructionError
from metrics.robustness import AdversarialRobustness, NoiseRobustness

np.random.seed(42)


def run_ssl(dataset_ids, transformers, clf, cv=3, labeled_splits=np.arange(0.05, 1.01, 0.05), score_metric='accuracy',
            scaling='minmax'):
    dfs = []
    for dataset_id in dataset_ids:
        for transformer in transformers:
            x, y = get_openml_data(dataset_id, scaling=scaling)
            skf = StratifiedKFold(cv)
            scorer = get_scorer(score_metric)
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
                    entry_name = f'split{fold}_test_{score_metric}'
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


def run_clf_test(dataset_ids, clfs, scaling='minmax', cv=3, score_metric='accuracy',
                 params=np.arange(0.25, 2.01, 0.25)):
    dfs = []
    for dataset_id in dataset_ids:
        for param in params:
            x, y = get_openml_data(dataset_id, subsample_size=10000, scaling=scaling)
            skf = StratifiedKFold(cv)
            scorer = get_scorer(score_metric)
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
                    results_scenario[f'split{fold}_test_{score_metric}'] = scorer(clf, x_test_encoded, y_test)
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
    pipe1 = Pipeline([
        ('trafo', PipelineHelper([
            ('ae', AutoTransformer()),
            ('none', IdentityTransformer())
        ])),
        ('clf', LogisticRegression(max_iter=500))
    ])
    params1 = {
        'trafo__selected_model': pipe1.named_steps['trafo'].generate({
            'ae__type': ['ae', 'vae', 'dae', 'sae']
        })
    }

    results = {}
    for dataset_id in dataset_ids:
        print(f'---------Dataset: {dataset_id}---------')
        x, y = get_openml_data(dataset_id, scaling=scaling)
        n_components = (np.arange(0.05, 1.01, 0.05) * np.prod(x.shape[1:])).astype(int)
        pipe = Pipeline([
            ('pca', PCA()),
            ('clf', LogisticRegression(penalty='none', max_iter=500))
        ])
        params = {
            'pca__n_components': n_components
        }
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


def run_training_robustness_test(dataset_ids, transformers, clfs, scaling='minmax', cv=3, score_metric='accuracy',
                                 noise_levels=np.arange(0, 0.51, 0.05), corrupt_type='snp'):
    results = {}
    for dataset_id in dataset_ids:
        for noise_level in noise_levels:
            x, y = get_openml_data(dataset_id, scaling=scaling, corrupt_type=corrupt_type, noise_level=noise_level)
            skf = StratifiedKFold(cv)
            scorer = get_scorer(score_metric)
            for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
                for transformer in transformers:
                    print(f'id:{dataset_id}, level:{noise_level}, fold: {fold}, transformer: {str(transformer)}')
                    start = time()
                    y_train, y_test = y[train_idx], y[test_idx]
                    x_train_encoded = transformer.fit_transform(x[train_idx])
                    x_test_encoded = transformer.transform(x[test_idx])
                    print(f'{str(transformer)}: fitting completed. Time: {round((time() - start) / 60, 2)} min.')
                    for clf in clfs:
                        start = time()
                        clf.fit(x_train_encoded, y_train)
                        score = scorer(clf, x_test_encoded, y_test)
                        key = (dataset_id, str(transformer), str(clf), noise_level)
                        if key in results:
                            results[key][f'split{fold}_test_{score_metric}'] = score
                        else:
                            results[key] = {f'split{fold}_test_{score_metric}': score}
                        print(f'{str(clf)}: fitting completed. Time: {round((time() - start) / 60, 2)} min.')
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.names = ['dataset_id', 'transformer', 'clf', 'noise_level']
    df = df.reset_index()
    return compute_means(df)


def run_testing_robustness_test(dataset_ids, transformers, clfs, scaling='minmax', cv=3, score_metric='accuracy',
                                noise_levels=np.arange(0, 0.51, 0.05), corrupt_types=('snp', 'gaussian'),
                                use_reconstructions=False):
    results = {}

    for dataset_id in dataset_ids:
        x, y = get_openml_data(dataset_id, scaling=None, corrupt_type=None)
        skf = StratifiedKFold(cv)
        scorer = get_scorer(score_metric)
        for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
            for transformer in transformers:
                print(f'id:{dataset_id}, fold: {fold}, transformer: {str(transformer)}')
                start = time()
                y_train, y_test = y[train_idx], y[test_idx]
                x_train, x_test = x[train_idx], x[test_idx]
                scaler = MinMaxScaler() if scaling == 'minmax' else StandardScaler
                x_train = scaler.fit_transform(x_train)
                x_train_encoded = transformer.fit_transform(x_train)
                if use_reconstructions:
                    x_train_encoded = transformer.inverse_transform(x_train_encoded)
                fit_time_transformer = time() - start
                print(f'{str(transformer)}: fitting completed. Time: {round(fit_time_transformer / 60, 2)} min.')
                for clf in clfs:
                    start = time()
                    clf.fit(x_train_encoded, y_train)
                    fit_time_clf = time() - start
                    print(f'{str(clf)}: fitting completed. Time: {round(fit_time_clf / 60, 2)} min.')
                    for noise_level in noise_levels:
                        for corrupt_type in corrupt_types:
                            corrupt = corrupt_snp if corrupt_type == 'snp' else corrupt_gaussian
                            start = time()
                            x_test_corrupted = corrupt(x_test, noise_level=noise_level)
                            x_test_corrupted = np.clip(scaler.transform(x_test_corrupted), a_min=0, a_max=1)
                            x_test_encoded = transformer.transform(x_test_corrupted)
                            if use_reconstructions:
                                x_test_encoded = transformer.inverse_transform(x_test_encoded)
                            score = scorer(clf, x_test_encoded, y_test)
                            key = (dataset_id, str(transformer), str(clf), noise_level, corrupt_type)
                            if key in results:
                                results[key][f'split{fold}_test_{score_metric}'] = score
                                results[key][f'split{fold}_fit_time_transformer'] = fit_time_transformer
                                results[key][f'split{fold}_fit_time_clf'] = fit_time_clf
                            else:
                                results[key] = {f'split{fold}_test_{score_metric}': score,
                                                f'split{fold}_fit_time_transformer': fit_time_transformer,
                                                f'split{fold}_fit_time_clf': fit_time_clf}
                        print(f'Level {noise_level} testing completed. Time: {round((time() - start) / 60, 2)} min.')
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.names = ['dataset_id', 'transformer', 'clf', 'noise_level', 'corrupt_type']
    df = df.reset_index()
    return compute_means(df)


def run_transformer_test(dataset_ids, transformers, scorers, clfs, scaling='minmax', cv=3):
    results = {}
    for dataset_id in dataset_ids:
        x, y = get_openml_data(dataset_id, scaling=scaling, corrupt_type=None)
        skf = StratifiedKFold(cv)
        for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            for transformer in transformers:
                print(f'id:{dataset_id}, fold: {fold}, transformer: {str(transformer)}')
                for clf in clfs:
                    key = (dataset_id, str(transformer), str(clf))
                    pipe = Pipeline([
                        ('trafo', transformer),
                        ('clf', clf)
                    ])
                    start = time()
                    pipe.fit(x_train, y_train)
                    results[key][f'split{fold}_fit_time'] = time() - start
                    for score_name, scorer in scorers.items():
                        score = scorer(pipe, x_test, y_test)
                        if key in results:
                            results[key][f'split{fold}_test_{score_name}'] = score
                        else:
                            results[key] = {f'split{fold}_test_{score_name}': score}

    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.names = ['dataset_id', 'transformer', 'clf']
    df = df.reset_index()
    return compute_means(df)


if __name__ == '__main__':
    datasets = [40996, 40668, 1492, 44]
    scorers = {'accuracy': get_scorer('accuracy')}
    clfs = [TPOTClassifier(generations=5, population_size=20, verbosity=2, cv=3, scoring="accuracy", n_jobs=50)]
    df = run_testing_robustness_test(datasets,
                                     transformers=[AutoTransformer(type='dae')],
                                     clfs=clfs)
    df.to_csv('tpot_w_ae.csv')
