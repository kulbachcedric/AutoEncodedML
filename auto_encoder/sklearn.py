from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf

from auto_encoder.model import Autoencoder, ConvAutoencoder


class SklearnTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, model_class, optimizer='adam', loss='mse', **ae_kwargs):
        self.optimizer = optimizer
        self.loss = loss
        self.model = None
        self.model_class = model_class
        self.ae_kwargs = ae_kwargs
        self.callback = self.callback = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=5)

    def fit(self, X, y=None, max_epochs=100, **kwargs):
        if self.model is None:
            self.model = self.model_class(**self.ae_kwargs)

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


class AutoTransformer(SklearnTransformer):
    def __init__(self, first_dim=0.6, latent_dim=0.2, n_layers=3, net_shape='geom', activation='selu', input_dropout=0,
                 dropout=0.5, regularizer=None, tied_weights=True, final_activation=None, loss='mse', optimizer='adam'):

        self.hidden_dims = (first_dim, latent_dim)
        super(AutoTransformer, self).__init__(model_class=Autoencoder, hidden_dims=self.hidden_dims, n_layers=n_layers,
                                              net_shape=net_shape, activation=activation, input_dropout=input_dropout,
                                              dropout=dropout, regularizer=regularizer, tied_weights=tied_weights,
                                              final_activation=final_activation, loss=loss, optimizer=optimizer)


class ConvAutoTransformer(SklearnTransformer):
    def __init__(self, filters_base=8, pooling=2, k_size=3, n_conv=2, strides=1, activation='relu', dropout=0.5,
                 latent_dim=0.05, loss='mse', optimizer='adam'):

        super(ConvAutoTransformer, self).__init__(filters_base=filters_base, pooling=pooling, k_size=k_size,
                                                  n_conv=n_conv, strides=strides, activation=activation,
                                                  dropout=dropout, latent_dim=latent_dim, loss=loss, optimizer=optimizer)
