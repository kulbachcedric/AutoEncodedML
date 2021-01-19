import numpy as np
import pandas as pd
from pipelinehelper import PipelineHelper
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, get_scorer
from xgboost import XGBClassifier
from datetime import datetime
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers as tfkl
from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd

from auto_encoder.sklearn import AutoTransformer
from data.openml import get_openml_data
from data.util import get_train_test_indices
from experiments.util import cv_results_to_df
from metrics.reconstruction import ReconstructionError
from metrics.robustness import AdversarialRobustness, NoiseRobustness
from auto_encoder.model import Autoencoder, VAE, get_vae

dataset_ids = [40996, 40668, 1492, 44]


def test_params(dataset_id, estimator, params, scorers, cv=1):
    x, y = get_openml_data(dataset_id, scale='minmax')
    splits = [get_train_test_indices(y)] if cv == 1 else cv
    grid = GridSearchCV(estimator=estimator, param_grid=params, cv=splits, scoring=scorers, refit=False)
    grid.fit(x, y)
    return cv_results_to_df(grid.cv_results_)


if __name__ == '__main__':
    np.random.seed(44)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.
    x_test = x_test.astype("float32") / 255.
    vae = Autoencoder(hidden_dims=2)
    vae.compile(loss='binary_crossentropy', optimizer='adam')
    vae.fit(x_train, x_train, epochs=10)
    x_encoded = vae.encoder(x_train)
    df = pd.DataFrame(x_encoded.numpy())
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(211)
    df[0].hist(ax=axs[0])
    df[1].hist(ax=axs[1])
    plt.show()
    """
    pipe = Pipeline([
        ('at', AutoTransformer(final_activation='sigmoid')),
        ('clf', PipelineHelper([('svm_rbf', SVC(max_iter=500)),
                                ('log_reg', LogisticRegression(max_iter=500)),
                                ('xgb', XGBClassifier())]))
    ])

    params = {
        'at__hidden_dims': [0.4],
        'at__activation': [None, 'selu', 'relu', 'tanh', 'sigmoid'],  # linear?
        'at__n_layers': [1, 2, 3, 4, 5],
        'clf__selected_model': pipe.named_steps['clf'].generate({
        })
    }
    scorers = {'accuracy': get_scorer('accuracy'), 'reconstruction_error': ReconstructionError()}
    np.random.seed(42)
    results = {}
    for dataset_id in dataset_ids:
        print(f'---------Dataset: {dataset_id}---------')
        # Autoencoder.cache_clear()
        results[dataset_id] = test_params(dataset_id=dataset_id, estimator=pipe, params=params, scorers=scorers, cv=3)
    df = pd.concat(results)
    df.index.names = ('dataset_id', 'idx')
    df.reset_index(level='idx', drop=True, inplace=True)
    day = datetime.now().strftime('%j')
    df.to_csv(f'gridsearchcv_results_{day}.csv')
    """


# Potential parameters for later tests
def optional_params():
    pipe = Pipeline([
        ('transformer', PipelineHelper([
            ('at', AutoTransformer(tied_weights=False, final_activation='sigmoid')),
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
