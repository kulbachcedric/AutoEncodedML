from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf

from auto_encoder.ae import Autoencoder
from auto_encoder.conv_ae import ConvAutoencoder
from auto_encoder.utils import calculate_hidden_dims


class AETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, input_shape, first_dim=100, n_layers=3, latent_dim=10, net_shape='geom',
                 activation='selu', dropout=0.5, regularizer=None,
                 tied_weights=True, final_activation=None, loss='mse', optimizer='adam'):

        self.first_dim = first_dim
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.activation = activation
        self.dropout = dropout
        self.regularizer = regularizer
        self.tied_weights = tied_weights
        self.final_activation = final_activation
        self.loss = loss
        self.optimizer = optimizer
        self.net_shape = net_shape
        self.callback = tf.keras.callbacks.EarlyStopping(min_delta=1e-4, patience=3)
        self.ae = None

    def fit(self, X, y=None, epochs=50):
        if not self.ae:
            hidden_dims = calculate_hidden_dims(self.first_dim, self.n_layers, self.latent_dim, self.net_shape)

            self.ae = Autoencoder(input_shape=self.input_shape,
                                  hidden_dims=hidden_dims,
                                  activation=self.activation,
                                  dropout=self.dropout,
                                  regularizer=self.regularizer,
                                  tied_weights=self.tied_weights,
                                  final_activation=self.final_activation)
            if not self.ae.is_fit:
                self.ae.compile(loss=self.loss, optimizer=self.optimizer)
                self.ae.fit(X, X, epochs=epochs, validation_split=0.1, callbacks=[self.callback])
                self.ae.is_fit = True
                tf.keras.backend.clear_session()
        return self

    def fit_transform(self, X, y=None, epochs=50):
        self.fit(X)
        return self.ae.encoder(X).numpy()

    def transform(self, X, y=None):
        return self.ae.encoder(X).numpy()


class CAETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, input_shape, filters_base=8, pooling=2, k_size=3, n_conv=2, strides=1,
                 activation='relu', dropout=0, latent_dim=20, loss='mse', optimizer='adam'):
        self.input_shape = input_shape
        self.filters_base = filters_base
        self.pooling = pooling
        self.k_size = k_size
        self.n_conv = n_conv
        self.strides = strides
        self.activation = activation
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.loss = loss
        self.optimizer = optimizer

        self.cae = ConvAutoencoder(self.input_shape, self.filters_base, self.pooling,
                                   self.k_size, self.n_conv, self.strides, self.activation,
                                   self.dropout, self.latent_dim)

        self.cae.compile(loss=self.loss, optimizer=self.optimizer)
        tf.keras.backend.clear_session()

    def fit(self, X, y=None, epochs=50):
        self.cae.fit(X, X, epochs=epochs, validation_split=0.1, callbacks=[self.callback])

    def fit_transform(self, X, y=None, epochs=50):
        self.cae.fit(X, X, epochs=epochs, validation_split=0.1, callbacks=[self.callback])
        return self.cae.encoder(X).numpy()

    def transform(self, X, y=None):
        return self.cae.encoder(X).numpy()
