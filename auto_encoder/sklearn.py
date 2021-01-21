from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from auto_encoder.model import *


class Transformer(BaseEstimator, TransformerMixin):
    models = {'ae': AE,
              'vae': VAE,
              'dae': DAE,
              'sae': SAE,
              'cae': CAE,
              'cvae': CVAE,
              'cdae': CDAE,
              'csae': CSAE}

    def __init__(self, type='ae', loss='binary_crossentropy', optimizer='adam', max_epochs=100, **ae_kwargs):
        self.type = type
        self.model = None
        self.loss = loss
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.ae_kwargs = ae_kwargs
        self.callback = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=2, restore_best_weights=True)

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1] if len(X.shape) == 2 else X.shape[1:]

        if isinstance(self.type, tf.keras.Model):
            self.model = self.type
        else:
            self.model = Transformer.models[self.type](**self.ae_kwargs)
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.model.fit(X, X, epochs=self.max_epochs, validation_split=0.1, callbacks=[self.callback])
        tf.keras.backend.clear_session()
        return self

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y)
        return self.model.encode(X).numpy()

    def transform(self, X, y=None):
        return self.model.encode(X).numpy()

    def inverse_transform(self, X):
        return self.model.decoder(X).numpy()


class AutoTransformer(Transformer):
    def __init__(self, type='ae', loss='binary_crossentropy', optimizer='adam', max_epochs=100, hidden_dims=0.2,
                 n_layers=3, activation='selu', output_activation='sigmoid'):
        self.type = type
        self.loss = loss
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.activation = activation
        self.output_activation = output_activation
        super(AutoTransformer, self).__init__(type=self.type, loss=self.loss, optimizer=self.optimizer,
                                              max_epochs=self.max_epochs, hidden_dims=self.hidden_dims,
                                              n_layers=self.n_layers, activation=self.activation,
                                              output_activation=self.output_activation)


class ConvolutionalAutoTransformer(Transformer):
    def __init__(self, type='cae', loss='binary_crossentropy', optimizer='adam', max_epochs=100, filters_base=8,
                 pooling=2, k_size=3, n_conv=2, strides=1, activation='selu', latent_dim=0.2,
                 output_activation='sigmoid'):
        super().__init__(type=type, loss=loss, optimizer=optimizer, max_epochs=max_epochs, filters_base=filters_base,
                         pooling=pooling, k_size=k_size, n_conv=n_conv, strides=strides, activation=activation,
                         latent_dim=latent_dim, output_activation=output_activation)


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return X

    def transform(self, X, y=None):
        return X
