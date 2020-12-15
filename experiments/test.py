from pipelinehelper import PipelineHelper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from auto_encoder.model import Autoencoder
from auto_encoder.sklearn import AutoTransformer, ConvAutoTransformer
from dataloader.openml import get_openml_data
from dataloader.util import get_train_test_indices
import pandas as pd
from experiments.util import cv_results_to_df
import numpy as np

dataset_ids = [40996, 40668, 1492, 44]

layer_sizes1 = [(1.0, 0.05), (1.0, 0.1), (1.0, 0.2), (1.0, 0.4), (1.0, 0.8)]

layer_sizes2 = [(0.1, 0.1), (0.2, 0.2), (0.4, 0.4), (0.8, 0.8), (1.6, 1.6)]

layer_sizes3 = [(0.1, 0.1), (0.2, 0.1), (0.4, 0.1), (0.8, 0.1), (1.6, 0.1)]

layer_sizes = [(1.0, 0.05), (1.0, 0.1), (1.0, 0.2), (1.0, 0.4), (1.0, 0.8),
               (0.1, 0.1), (0.2, 0.1), (0.4, 0.1), (0.8, 0.1), (1.6, 0.1)]

layers = [(1, 2, 3, 4, 5)]
activations = ['selu', 'relu', 'tanh', 'sigmoid']


def test_dataset(dataset_id):
    x, y = get_openml_data(dataset_id, scale='minmax')
    split_indices = get_train_test_indices(y)

    pipe = Pipeline([
        ('at', AutoTransformer(final_activation='sigmoid')),
        ('clf', PipelineHelper([('svm_rbf', SVC(max_iter=500)),
                                ('log_reg', LogisticRegression(max_iter=500)),
                                ('xgb', XGBClassifier())]))
    ])

    params = {
        'at__hidden_dims': layer_sizes,
        'at__dropout': [(0, 0), (0.2, 0.2), (0.2,0.5)],
        'at__activation': ['selu'],
        'at__n_layers': [2, 4],
        'clf__selected_model': pipe.named_steps['clf'].generate({
        })
    }

    grid = GridSearchCV(pipe, param_grid=params, cv=[split_indices])
    grid.fit(x, y)
    return cv_results_to_df(grid.cv_results_)


def calculate_hidden_dims(net_shape, hidden_dims, n_layers, input_shape):

    dim_calculator = np.linspace if net_shape == 'linear' else np.geomspace

    if isinstance(hidden_dims, float):
        dims = hidden_dims * np.prod(input_shape)
    elif isinstance(hidden_dims, (tuple, list)):
        dims = [dim * np.prod(input_shape) if isinstance(dim, float) else dim for dim in hidden_dims]
    else:
        dims = hidden_dims

    if isinstance(dims, (tuple, list)) and len(dims) == 2:
        dims = dim_calculator(*dims, num=n_layers)
    elif isinstance(dims, (int, float)):
        dims = dim_calculator(np.prod(input_shape), dims, num=n_layers + 1)[1:]

    dims = np.round(dims).astype(int)
    return tuple(dims)


if __name__ == '__main__':
    results = {}
    for dataset_id in dataset_ids:
        print(f'---------Dataset: {dataset_id}---------')
        Autoencoder.cache_clear()
        results[dataset_id] = test_dataset(dataset_id)
    df = pd.DataFrame(results)
    df.index.names = ('dataset_id', 'idx')
    df.reset_index(level='idx', drop=True, inplace=True)
    df.to_csv('gridsearch_results.csv')

