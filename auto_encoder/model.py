import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from auto_encoder.component import DenseTranspose


@functools.lru_cache(maxsize=10)
class Autoencoder(tf.keras.Model):

    def __init__(self, hidden_dims=(0.6, 0.2), n_layers=3, net_shape='geom', activation='selu', input_dropout=0,
                 dropout=0.5, regularizer=None, tied_weights=True, final_activation=None):

        super().__init__()
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.net_shape = net_shape
        self.activation = activation
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.regularizer = regularizer
        self.tied_weights = tied_weights
        self.final_activation = final_activation
        self.layers_encoder = []
        self.layers_decoder = []
        self.encoder = None
        self.decoder = None
        self.is_fit = False

    def build(self, input_shape):
        input_shape = input_shape[1:]
        self.hidden_dims_abs = [dim * np.prod(input_shape) if isinstance(dim, float) else dim for dim in
                                self.hidden_dims]

        if len(self.hidden_dims_abs) == 2 and self.n_layers > 2:
            self.hidden_dims_abs = self.calculate_hidden_dims()

        for dim in self.hidden_dims_abs[:-1]:
            self.layers_encoder.append(layers.Dense(dim, activation=self.activation))
            if self.dropout is not None and self.dropout > 0:
                self.layers_encoder.append(layers.Dropout(self.dropout))
        self.layers_encoder.append(
            layers.Dense(self.hidden_dims_abs[-1], self.activation, activity_regularizer=self.regularizer))

        if self.tied_weights:
            for index, layer in enumerate(self.layers_encoder[::-1]):
                if isinstance(layer, layers.Dense):
                    if index >= len(self.layers_encoder) - 1:
                        self.layers_decoder.append(DenseTranspose(layer, activation=self.final_activation))
                    else:
                        self.layers_decoder.append(DenseTranspose(layer, activation=self.activation))
        else:
            for dim in self.hidden_dims_abs[1::-1]:
                self.layers_decoder.append(layers.Dense(dim, activation=self.activation))
            self.layers_decoder.append(layers.Dense(np.prod(input_shape), activation=self.final_activation))

        if isinstance(input_shape, tuple) and len(input_shape) > 1:
            self.layers_decoder.append(layers.Reshape(input_shape))
            self.layers_encoder.insert(0, layers.Flatten(input_shape=input_shape))

        if self.input_dropout is not None and self.input_dropout > 0:
            self.layers_encoder.insert(0, layers.Dropout(self.input_dropout))

        self.encoder = tf.keras.Sequential(self.layers_encoder)
        self.decoder = tf.keras.Sequential(self.layers_decoder)

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def calculate_hidden_dims(self):
        if self.net_shape == 'linear':
            dims = np.linspace(*self.hidden_dims_abs, self.n_layers + 1)
        else:
            dims = np.geomspace(*self.hidden_dims_abs, self.n_layers + 1)

        dims = np.round(dims[1:]).astype(int)
        return tuple(dims)


class ConvAutoencoder(tf.keras.Model):
    def __init__(self, filters_base=8, pooling=2, k_size=3, n_conv=2, strides=1,
                 activation='relu', dropout=0.5, latent_dim=0.05):

        super(ConvAutoencoder, self).__init__()
        self.filters_base = filters_base
        self.pooling = pooling
        self.n_conv = n_conv
        self.strides = strides
        self.activation = activation
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        if isinstance(k_size, (int, np.int)):
            self.k_size = np.repeat(k_size, self.n_conv)
        else:
            self.k_size = k_size

    def build(self, input_shape):
        if isinstance(self.latent_dim, float):
            self.latent_dim = np.round(self.latent_dim * np.prod(input_shape)).astype(int)
        layers_encoder = [layers.Input(input_shape)]
        layers_decoder = []
        self.layers_encoder.append(layers.Dropout(0.2))

        for i in range(self.n_conv):
            k_size_i = (self.k_size[i], self.k_size[i])
            layers_encoder.append(
                layers.Conv2D(self.filters_base * 2 ** i, kernel_size=k_size_i,
                              activation=self.activation, padding='same',
                              strides=self.strides))

            layers_decoder.append(
                layers.Conv2DTranspose(self.filters_base * 2 ** (self.n_conv - i - 1),
                                       kernel_size=k_size_i, activation=self.activation,
                                       padding='same', strides=self.strides))

            if self.pooling and self.pooling > 0:
                layers_encoder.append(layers.MaxPool2D(pool_size=self.pooling))
                layers_decoder.append(layers.UpSampling2D(size=self.pooling))

        self.encoder = tf.keras.Sequential(layers_encoder)

        encoder_output = self.encoder.compute_output_shape((None,) + input_shape)
        self.encoder.add(layers.Flatten())

        if self.dropout and self.dropout > 0:
            self.encoder.add(layers.Dropout(self.dropout))

        bottleneck = layers.Dense(self.latent_dim, activation=self.activation)
        self.encoder.add(bottleneck)

        layers_decoder[0:0] = [DenseTranspose(bottleneck, activation=self.activation),
                               layers.Reshape(encoder_output[1:])]
        self.decoder = tf.keras.Sequential(layers_decoder)
        decoder_output = self.decoder.compute_output_shape((None,) + (self.latent_dim,))

        self.decoder.add(
            layers.Conv2DTranspose(input_shape[-1], input_shape[1] + 1 - decoder_output[1], activation='sigmoid'))

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
