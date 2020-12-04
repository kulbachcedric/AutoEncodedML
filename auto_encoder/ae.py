import functools
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from auto_encoder.utils import calculate_hidden_dims
from classifier.utils import DenseTranspose


@functools.lru_cache(maxsize=10)
class Autoencoder(tf.keras.Model):

    def __init__(self, input_shape, hidden_dims, activation='selu', input_dropout=0,
                 dropout=0.5, regularizer=None, tied_weights=True, final_activation=None):

        super().__init__()
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.regularizer = regularizer
        self.tied_weights = tied_weights
        self.layers_encoder = []
        self.layers_decoder = []
        self.final_activation = final_activation
        self.is_fit = False


        for dim in self.hidden_dims:
            self.layers_encoder.append(layers.Dense(dim, activation=self.activation))
            if self.dropout > 0:
                self.layers_encoder.append(layers.Dropout(self.dropout))
        self.layers_encoder.append(
            layers.Dense(self.hidden_dims[-1], self.activation, activity_regularizer=self.regularizer))

        if self.tied_weights:
            for index, layer in enumerate(self.layers_encoder[::-1]):
                if isinstance(layer, layers.Dense):
                    if index >= len(self.layers_encoder) - 1:
                        self.layers_decoder.append(DenseTranspose(layer, activation=self.final_activation))
                    else:
                        self.layers_decoder.append(DenseTranspose(layer, activation=self.activation))

        else:
            for dim in self.hidden_dims[::-1]:
                self.layers_decoder.append(layers.Dense(dim, activation=self.activation))
            self.layers_decoder.append(layers.Dense(np.prod(input_shape), activation=self.final_activation))

        if type(input_shape) is tuple and len(input_shape) > 1:
            self.layers_decoder.append(layers.Reshape(input_shape))
            self.layers_encoder[0:0] = [layers.Flatten(input_shape=input_shape)]

        if self.input_dropout > 0:
            self.layers_encoder[0:0] = [layers.Dropout(self.input_dropout)]

        self.encoder = tf.keras.Sequential(self.layers_encoder)
        self.decoder = tf.keras.Sequential(self.layers_decoder)

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
