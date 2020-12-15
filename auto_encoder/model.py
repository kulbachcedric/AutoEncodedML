import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from auto_encoder.component import DenseTranspose


@functools.lru_cache(maxsize=5)
class Autoencoder(tf.keras.Model):

    def __init__(self, hidden_dims=(0.6, 0.2), n_layers=3, net_shape='geom', activation='selu', dropout=(0.2, 0.5),
                 regularizer=None, tied_weights=True, final_activation=None):

        super().__init__()
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.net_shape = net_shape
        self.activation = activation
        self.dropout = dropout
        self.regularizer = regularizer
        self.tied_weights = tied_weights
        self.final_activation = final_activation
        self.encoder = None
        self.decoder = None
        self.is_fit = False

    def build(self, input_shape):
        layers_encoder = []
        layers_decoder = []
        input_shape = input_shape[1:]
        self.hidden_dims_abs = self.calculate_hidden_dims(hidden_dims=self.hidden_dims, n_layers=self.n_layers,
                                                          input_shape=input_shape)

        if not isinstance(self.dropout, tuple) or len(self.dropout) < 2:
            input_dropout = self.dropout
            hidden_dropout = None
        else:
            input_dropout, hidden_dropout = self.dropout

        if input_dropout and input_dropout > 0:
            layers_encoder.insert(0, layers.Dropout(input_dropout))

        for dim in self.hidden_dims_abs[:-1]:
            layers_encoder.append(layers.Dense(dim, activation=self.activation))
            if hidden_dropout and hidden_dropout > 0:
                layers_encoder.append(layers.Dropout(hidden_dropout))
        layers_encoder.append(
            layers.Dense(self.hidden_dims_abs[-1], self.activation, activity_regularizer=self.regularizer))

        if self.tied_weights:
            dense_layers = [layer for layer in layers_encoder[::-1] if isinstance(layer, layers.Dense)]
            for layer in dense_layers[:-1]:
                layers_decoder.append(DenseTranspose(layer, activation=self.activation))
            layers_decoder.append(DenseTranspose(dense_layers[-1], activation=self.final_activation))
        else:
            for dim in self.hidden_dims_abs[1::-1]:
                layers_decoder.append(layers.Dense(dim, activation=self.activation))
            layers_decoder.append(layers.Dense(np.prod(input_shape), activation=self.final_activation))

        if isinstance(input_shape, tuple) and len(input_shape) > 1:
            layers_decoder.append(layers.Reshape(input_shape))
            layers_encoder.insert(0, layers.Flatten(input_shape=input_shape))

        self.encoder = tf.keras.Sequential(layers_encoder)
        self.decoder = tf.keras.Sequential(layers_decoder)

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def calculate_hidden_dims(self, hidden_dims, n_layers, input_shape):

        dim_calculator = np.linspace if self.net_shape == 'linear' else np.geomspace

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


@functools.lru_cache(maxsize=5)
class ConvAutoencoder(tf.keras.Model):
    def __init__(self, filters_base=8, pooling=2, k_size=3, n_conv=2, strides=1,
                 activation='relu', dropout=0.5, latent_dim=0.05, regularizer=None):

        super(ConvAutoencoder, self).__init__()
        self.filters_base = filters_base
        self.pooling = pooling
        self.n_conv = n_conv
        self.strides = strides
        self.activation = activation
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.regularizer = regularizer
        self.encoder = None
        self.decoder = None
        self.is_fit = False
        if isinstance(k_size, (int, np.int)):
            self.k_size = np.repeat(k_size, self.n_conv)
        else:
            self.k_size = k_size

    def build(self, input_shape):
        input_shape = input_shape[1:]
        if isinstance(self.latent_dim, float):
            self.latent_dim = np.round(self.latent_dim * np.prod(input_shape)).astype(int)
        layers_encoder = [layers.Input(input_shape)]
        layers_decoder = []

        if not isinstance(self.dropout, tuple) or len(self.dropout) < 2:
            input_dropout = self.dropout
            hidden_dropout = None
        else:
            input_dropout, hidden_dropout = self.dropout

        if input_dropout and input_dropout > 0:
            layers_encoder.append(layers.Dropout(input_dropout))

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

        if hidden_dropout and hidden_dropout > 0:
            self.encoder.add(layers.Dropout(hidden_dropout))

        bottleneck = layers.Dense(self.latent_dim, activation=self.activation, activity_regularizer=self.regularizer)
        self.encoder.add(bottleneck)

        decoder_output = encoder_output
        for layer in layers_decoder:
            decoder_output = layer.compute_output_shape(decoder_output)

        layers_decoder[0:0] = [DenseTranspose(bottleneck, activation=self.activation),
                               layers.Reshape(encoder_output[1:])]

        layers_decoder.append(
            layers.Conv2DTranspose(input_shape[-1], input_shape[1] + 1 - decoder_output[1], activation='sigmoid'))
        self.decoder = tf.keras.Sequential(layers_decoder)

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
