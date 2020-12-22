from pipelinehelper import PipelineHelper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from xgboost import XGBClassifier

from auto_encoder.model import Autoencoder
from auto_encoder.sklearn import AutoTransformer, IdentityTransformer
from data.openml import get_openml_data
from data.util import get_train_test_indices
import pandas as pd
from experiments.util import cv_results_to_df
import numpy as np

dataset_ids = [40996, 40668, 1492, 44]

layer_sizes = [(1.0, 0.05), (1.0, 0.1), (1.0, 0.2), (1.0, 0.4), (1.0, 0.8),
               (0.1, 0.1), (0.2, 0.1), (0.4, 0.1), (0.8, 0.1), (1.6, 0.1)]


def test_params(dataset_id):
    x, y = get_openml_data(dataset_id, scale='minmax')
    split_indices = get_train_test_indices(y)

    pipe = Pipeline([
        ('at', AutoTransformer(final_activation='sigmoid')),
        ('clf', PipelineHelper([('svm_rbf', SVC(max_iter=500)),
                                ('log_reg', LogisticRegression(max_iter=500)),
                                ('xgb', XGBClassifier())]))
    ])

    params = {
        'at__hidden_dims': np.arange(0.05, 2.05, 0.05),
        'at__activation': ['selu'],
        'at__n_layers': [2, 4],
        'clf__selected_model': pipe.named_steps['clf'].generate({
        })
    }

    # Test for network depth and activation function
    params2 = {
        'at__hidden_dims': [0.4],
        'at__dropout': [(0, 0), (0.2, 0.2), (0.2, 0.5)],
        'at__activation': ['selu', 'relu', 'tanh', 'sigmoid'],  # linear?
        'at__n_layers': [1, 2, 3, 4, 5],
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

    # Test for more detailed info on latent dimension
    params4 = {
        'at__hidden_dims': np.arange(0.1, 2.1, 0.1),
        'at__dropout': [(0, 0), (0.2, 0.2)],
        'at__activation': ['selu'],  # linear?
        'at__n_layers': [3],
        'clf__selected_model': pipe.named_steps['clf'].generate({
        })
    }

    # Test for more classification models/parameters?

    grid = GridSearchCV(pipe, param_grid=params, cv=[split_indices])
    grid.fit(x, y)
    return cv_results_to_df(grid.cv_results_)


def test_transformers(dataset_id):
    x, y = get_openml_data(dataset_id, scale='minmax')
    split_indices = get_train_test_indices(y)

    pipe = Pipeline([
        ('transformer', PipelineHelper([
            ('at', AutoTransformer(final_activation='sigmoid')),
            ('pca', PCA()),
        ])),
        ('clf', PipelineHelper([
            ('svm_rbf', SVC(max_iter=500)),
            ('log_reg', LogisticRegression(max_iter=500)),
        ]))
    ])

    params = {
        'transformer__selected_model': pipe.named_steps['transformer'].generate({
            'at__hidden_dims': [5],
            'pca__n_components': [5],
        }),
        'clf__selected_model': pipe.named_steps['clf'].generate({
        })
    }

    grid = GridSearchCV(pipe, param_grid=params, cv=[split_indices])
    grid.fit(x, y)
    return cv_results_to_df(grid.cv_results_, cols_expand=['clf', 'transformer'])


if __name__ == '__main__':
    results = {}
    for dataset_id in dataset_ids:
        print(f'---------Dataset: {dataset_id}---------')
        # Autoencoder.cache_clear()
        results[dataset_id] = test_params(dataset_id)
    df = pd.concat(results)
    df.index.names = ('dataset_id', 'idx')
    df.reset_index(level='idx', drop=True, inplace=True)
    df.to_csv('gridsearch_results.csv')
