from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from tensorflow import keras
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
                 n_layers=3, activation='selu', output_activation='sigmoid', dropout=None, hidden_dropout=None):
        super(AutoTransformer, self).__init__(type, loss, optimizer, max_epochs)
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.activation = activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.hidden_dropout = hidden_dropout

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1] if len(X.shape) == 2 else X.shape[1:]

        if isinstance(self.type, tf.keras.Model):
            self.model = self.type
        else:
            self.model = Transformer.models[self.type](self.hidden_dims, self.n_layers, self.activation,
                                                       self.output_activation, dropout=self.dropout, hidden_dropout=self.hidden_dropout)
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.model.fit(X, X, epochs=self.max_epochs, validation_split=0.1, callbacks=[self.callback])
        tf.keras.backend.clear_session()
        return self


class ConvolutionalAutoTransformer(Transformer):
    def __init__(self, type='cae', loss='binary_crossentropy', optimizer='adam', max_epochs=100, filters_base=8,
                 pooling=2, k_size=3, n_conv=2, strides=1, activation='selu', latent_dim=0.2,
                 output_activation='sigmoid'):
        super().__init__(type=type, loss=loss, optimizer=optimizer, max_epochs=max_epochs)
        self.filters_base = filters_base
        self.pooling = pooling
        self.k_size = k_size
        self.n_conv = n_conv
        self.strides = strides
        self.activation = activation
        self.latent_dim = latent_dim
        self.output_activation = output_activation

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1] if len(X.shape) == 2 else X.shape[1:]

        if isinstance(self.type, tf.keras.Model):
            self.model = self.type
        else:
            self.model = Transformer.models[self.type](self.filters_base,
                                                       self.pooling,
                                                       self.k_size,
                                                       self.n_conv,
                                                       self.strides,
                                                       self.activation,
                                                       self.latent_dim,
                                                       self.output_activation)
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.model.fit(X, X, epochs=self.max_epochs, validation_split=0.1, callbacks=[self.callback])
        tf.keras.backend.clear_session()
        return self


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return X

    def transform(self, X, y=None):
        return X


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_epochs=10000, batch_size=128, optimizer='adam', learning_rate=0.001):
        self.model = None
        self.callback = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=2, restore_best_weights=True)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def fit(self, X, y):
        input_shape = X.shape[1:]
        softmax = keras.layers.Dense(np.unique(y).size, activation='softmax')
        self.model = keras.Sequential([keras.layers.InputLayer(input_shape),
                                       softmax])
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy')
        self.model.fit(X, y, epochs=self.max_epochs, validation_split=0.1, callbacks=[self.callback],
                       batch_size=self.batch_size)
        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return tf.argmax(probs, axis=-1)

    def predict_proba(self, X):
        return self.model(X)
