from datetime import datetime

import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from auto_encoder.model import *
from auto_encoder.sklearn import AutoTransformer, SoftmaxClassifier
from data.openml import get_openml_data
from data.util import get_train_test_indices
from experiments.util import cv_results_to_df, remove_col_prefix
from metrics.reconstruction import ReconstructionError
from metrics.robustness import AdversarialRobustness, NoiseRobustness

np.random.seed(42)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def test_params(dataset_id, estimator, params, scorers, cv=1):
    x, y = get_openml_data(dataset_id, scale='minmax')
    splits = [get_train_test_indices(y)] if cv == 1 else cv
    grid = GridSearchCV(estimator=estimator, param_grid=params, cv=splits, scoring=scorers, refit=False)
    grid.fit(x, y)
    return cv_results_to_df(grid.cv_results_)


def run_gridsearch(dataset_ids, data_scaling='minmax', cv=3):
    pipe = Pipeline([
        ('ae', AutoTransformer()),
        ('clf', LogisticRegression(penalty='none', max_iter=500))
    ])
    params = {
        'clf__multi_class': ['ovr']
    }
    scorers = {'accuracy': get_scorer('accuracy'), 'reconstruction_error': ReconstructionError()}
    results = {}
    for dataset_id in dataset_ids:
        print(f'---------Dataset: {dataset_id}---------')
        x, y = get_openml_data(dataset_id, scale=data_scaling)
        splits = [get_train_test_indices(y)] if cv == 1 else cv
        grid = GridSearchCV(estimator=pipe, param_grid=params, cv=splits, scoring=scorers, refit=False, verbose=2)
        grid.fit(x, y)
        results[dataset_id] = pd.DataFrame(grid.cv_results_)

    df = pd.concat(results)
    df.index.names = ('dataset_id', 'idx')
    df.reset_index(level='idx', drop=True, inplace=True)
    day = datetime.now().strftime('%j')
    df.to_csv(f'gridsearchcv_results_{day}.csv')
    df = remove_col_prefix(df)
    df.to_csv(f'gridsearchcv_results_{day}.csv')


def test_classifiers(dataset_id):
    x, y = get_openml_data(dataset_id, scale='minmax')
    idx_train, idx_test = get_train_test_indices(y)
    at = AutoTransformer()
    x_encoded = at.fit_transform(x)
    clf = SoftmaxClassifier(max_epochs=5000)
    params = {
        'learning_rate': [1e-4, 1e-3, 1e-2]
    }
    grid = GridSearchCV(estimator=clf, param_grid=params, cv=[(idx_train, idx_test)], refit=False, verbose=2)
    grid.fit(x_encoded, y)
    return pd.DataFrame(grid.cv_results_)


if __name__ == '__main__':
    datasets = [40996, 40668, 1492, 44]
    run_gridsearch(datasets, cv=3)


# Potential parameters for later tests
def optional_params():
    pipe = Pipeline([
        ('ae', AutoTransformer()),
        ('clf', LogisticRegression(max_iter=500, penalty='none', solver='sag'))
    ])
    params = {
        'ae__type': ['ae'],
        'ae__hidden_dims': [np.arange(0.05, 2.05, 0.05)]
    }

    # Test for different transformators
    params_trafos = {
        'transformer__selected_model': pipe.named_steps['transformer'].generate({
            'at__hidden_dims': [5],
            'pca__n_components': [5],
        }),
        'clf__selected_model': pipe.named_steps['clf'].generate({
        })
    }

    # Test for network depth and activation function
    params2 = {
        'at__hidden_dims': [0.4],
        'at__activation': [None, 'selu', 'relu', 'tanh', 'sigmoid'],  # linear?
        'at__n_layers': [1, 2, 3, 4, 5],
        'clf__selected_model': pipe.named_steps['clf'].generate({
        })
    }

    # Test for latent dim
    params = {
        'at__hidden_dims': np.arange(0.05, 2.05, 0.05),
        'at__activation': ['selu'],
        'at__n_layers': [3],
        'clf__selected_model': pipe.named_steps['clf'].generate({
        })
    }

    # Test for dropout rate
    params3 = {
        'at__hidden_dims': [0.4],
        'at__dropout': [(0, 0), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)],
        'at__activation': ['selu'],  # linear?
        'at__n_layers': [3],
        'clf__selected_model': pipe.named_steps['clf'].generate({
        })
    }

    scorers_dict = {'accuracy': get_scorer('accuracy'),
                    'reconstruction_error': ReconstructionError(),
                    'adversarial_robustness': AdversarialRobustness(),
                    'noise_robustness': NoiseRobustness()}
