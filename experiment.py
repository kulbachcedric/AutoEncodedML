import numpy as np
import pandas as pd
from pipelinehelper import PipelineHelper
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, get_scorer
from datetime import datetime
import tensorflow as tf

from auto_encoder.sklearn import AutoTransformer, Transformer
from data.openml import get_openml_data
from data.util import get_train_test_indices
from experiments.util import cv_results_to_df
from metrics.reconstruction import ReconstructionError
from metrics.robustness import AdversarialRobustness, NoiseRobustness
from auto_encoder.model import *


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True) #TODO: !!!!CHECK GPU!!!!
np.random.seed(42)


def test_params(dataset_id, estimator, params, scorers, cv=1):
    x, y = get_openml_data(dataset_id, scale='minmax')
    splits = [get_train_test_indices(y)] if cv == 1 else cv
    grid = GridSearchCV(estimator=estimator, param_grid=params, cv=splits, scoring=scorers, refit=False)
    grid.fit(x, y)
    return cv_results_to_df(grid.cv_results_)


def run_gridsearch(dataset_ids):

    pipe = Pipeline([
        ('ae', AutoTransformer()),
        ('clf', LogisticRegression(max_iter=1000, penalty='none', solver='sag'))
    ])
    params = {
        'ae__type': ['ae', 'vae', 'dae', 'sae'],
        'ae__hidden_dims': np.arange(0.05, 2.05, 0.05)
    }
    scorers = {'accuracy': get_scorer('accuracy'), 'reconstruction_error': ReconstructionError()}
    results = {}
    for dataset_id in dataset_ids:
        print(f'---------Dataset: {dataset_id}---------')
        x, y = get_openml_data(dataset_id, scale='minmax')
        grid = GridSearchCV(estimator=pipe, param_grid=params, cv=3, scoring=scorers, refit=False)
        grid.fit(x, y)
        results[dataset_id] = pd.DataFrame(grid.cv_results_)

    df = pd.concat(results)
    df.index.names = ('dataset_id', 'idx')
    df.reset_index(level='idx', drop=True, inplace=True)
    day = datetime.now().strftime('%j')
    df.to_csv(f'gridsearchcv_results_{day}.csv')


if __name__ == '__main__':
    datasets = [40996, 40668, 1492, 44]
    run_gridsearch(datasets)


# Potential parameters for later tests
def optional_params():
    pipe = Pipeline([
        ('transformer', PipelineHelper([
            ('at', AutoTransformer()),
            ('pca', PCA()),
        ])),
        ('clf', PipelineHelper([
            ('svm_rbf', SVC(max_iter=500)),
            ('log_reg', LogisticRegression(max_iter=500)),
        ]))
    ])

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
