from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from auto_encoder.model import Autoencoder, ConvAutoencoder


class AutoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, hidden_dims=0.2, n_layers=3, net_shape='geom', activation='selu', dropout=None, regularizer=None,
                 tied_weights=True, final_activation='sigmoid', loss='mse', optimizer='adam', max_epochs=100):
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.net_shape = net_shape
        self.activation = activation
        self.dropout = dropout
        self.regularizer = regularizer
        self.tied_weights = tied_weights
        self.final_activation = final_activation
        self.loss = loss
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.model = None
        self.callback = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=5, restore_best_weights=True)

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1] if len(X.shape) == 2 else X.shape[1:]
        if self.model is None:
            self.model = Autoencoder(hidden_dims=self.hidden_dims, n_layers=self.n_layers, net_shape=self.net_shape,
                                     activation=self.activation, dropout=self.dropout, regularizer=self.regularizer,
                                     tied_weights=self.tied_weights, final_activation=self.final_activation)

        if not self.model.is_fit:
            self.model.compile(loss=self.loss, optimizer=self.optimizer)
            self.model.fit(X, X, epochs=self.max_epochs, validation_split=0.1, callbacks=[self.callback])
            self.model.is_fit = True
            tf.keras.backend.clear_session()

        return self

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y)
        return self.model.encoder(X).numpy()

    def transform(self, X, y=None):
        return self.model.encoder(X).numpy()

    def inverse_transform(self, X):
        return self.model.decoder(X).numpy()


class ConvAutoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, filters_base=8, pooling=2, k_size=3, n_conv=2, strides=1, activation='relu', dropout=(0.2, 0.5),
                 latent_dim=0.05, regularizer=None, loss='mse', optimizer='adam'):
        self.filters_base = filters_base
        self.pooling = pooling
        self.k_size = k_size
        self.n_conv = n_conv
        self.strides = strides
        self.activation = activation
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.regularizer = regularizer
        self.loss = loss
        self.optimizer = optimizer
        self.model = None
        self.callback = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=5, restore_best_weights=True)

    def fit(self, X, y=None, max_epochs=100):
        if self.model is None:
            self.model = ConvAutoencoder(filters_base=self.filters_base,
                                         pooling=self.pooling,
                                         k_size=self.k_size,
                                         n_conv=self.n_conv,
                                         strides=self.strides,
                                         activation=self.activation,
                                         dropout=self.dropout,
                                         latent_dim=self.latent_dim,
                                         regularizer=self.regularizer)

        if not self.model.is_fit:
            self.model.compile(loss=self.loss, optimizer=self.optimizer)
            self.model.fit(X, X, epochs=max_epochs, validation_split=0.1, callbacks=[self.callback])
            self.model.is_fit = True
            tf.keras.backend.clear_session()

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.model.encoder(X).numpy()

    def transform(self, X, y=None):
        return self.model.encoder(X).numpy()


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return X

    def transform(self, X, y=None):
        return X
