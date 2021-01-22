import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from auto_encoder.component import DenseTranspose, Sampling, LatentLossRegularizer, KLDRegularizer


class AE(tf.keras.Model):
    def __init__(self, hidden_dims=0.2, n_layers=3, activation='selu', tied_weights=True, dropout=None,
                 regularizer=None, output_activation='sigmoid'):

        super().__init__()
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.regularizer = regularizer
        self.tied_weights = tied_weights
        self.output_activation = output_activation
        self.encoder = None
        self.decoder = None
        self.latent_dim = None

    def build(self, input_shape):
        input_shape = input_shape[1:]
        self.hidden_dims = self.calculate_hidden_dims(hidden_dims=self.hidden_dims, n_layers=self.n_layers,
                                                      input_shape=input_shape)
        self.get_encoder_decoder(input_shape)

    def get_encoder_decoder(self, input_shape, latent_activation='sigmoid'):
        layers_encoder = []
        layers_decoder = []
        if self.dropout and self.dropout > 0:
            layers_encoder.append(layers.Dropout(self.dropout))

        for dim in self.hidden_dims[:-1]:
            layers_encoder.append(layers.Dense(dim, activation=self.activation))

        layers_encoder.append(
            layers.Dense(self.hidden_dims[-1], latent_activation, activity_regularizer=self.regularizer))

        if self.tied_weights:
            dense_layers = [layer for layer in layers_encoder[::-1] if isinstance(layer, layers.Dense)]
            for layer in dense_layers[:-1]:
                layers_decoder.append(DenseTranspose(layer, activation=self.activation))
            layers_decoder.append(DenseTranspose(dense_layers[-1], activation=self.output_activation))
        else:
            for dim in self.hidden_dims[1::-1]:
                layers_decoder.append(layers.Dense(dim, activation=self.activation))
            layers_decoder.append(layers.Dense(np.prod(input_shape), activation=self.output_activation))

        if not isinstance(input_shape, int) and len(input_shape) > 1:
            layers_decoder.append(layers.Reshape(input_shape))
            layers_encoder.insert(0, layers.Flatten(input_shape=input_shape))

        self.encoder = tf.keras.Sequential(layers_encoder, name='Encoder')
        self.decoder = tf.keras.Sequential(layers_decoder, name='Decoder')

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def calculate_hidden_dims(self, hidden_dims, n_layers, input_shape):
        if isinstance(hidden_dims, float):
            hidden_dims = hidden_dims * np.prod(input_shape)
        elif isinstance(hidden_dims, (tuple, list)):
            hidden_dims = [dim * np.prod(input_shape) if isinstance(dim, float) else dim for dim in hidden_dims]

        if isinstance(hidden_dims, (int, float)):
            hidden_dims = np.geomspace(np.prod(input_shape), hidden_dims, num=n_layers + 1)[1:]
        elif isinstance(hidden_dims, (tuple, list)):
            self.n_layers = len(hidden_dims)

        hidden_dims = np.round(hidden_dims).astype(int)
        self.latent_dim = hidden_dims[-1]
        return list(hidden_dims)

    def encode(self, x):
        return self.encoder(x)


class DAE(AE):
    def __init__(self, hidden_dims=0.2, n_layers=3, activation='selu', output_activation='sigmoid'):
        super(DAE, self).__init__(hidden_dims, n_layers=n_layers, activation=activation,
                                  output_activation=output_activation, dropout=0.2)


class SAE(AE):
    def __init__(self, hidden_dims=0.2, n_layers=3, activation='selu', output_activation='sigmoid'):
        super(SAE, self).__init__(hidden_dims, n_layers=n_layers, activation=activation,
                                  output_activation=output_activation, regularizer=KLDRegularizer(0.05),
                                  tied_weights=False)


class VAE(AE):
    def __init__(self, hidden_dims=0.2, n_layers=3, activation='selu', output_activation='sigmoid'):
        super().__init__(hidden_dims, n_layers=n_layers, activation=activation, output_activation=output_activation,
                         tied_weights=False, dropout=None)
        self.sampling = Sampling()

    def build(self, input_shape):
        input_shape = input_shape[1:]
        self.regularizer = LatentLossRegularizer(1 / np.prod(input_shape))
        self.hidden_dims = self.calculate_hidden_dims(hidden_dims=self.hidden_dims, n_layers=self.n_layers,
                                                      input_shape=input_shape)
        self.hidden_dims[-1] *= 2
        self.get_encoder_decoder(input_shape, latent_activation='linear')

    def call(self, x, training=None, mask=None):
        encoded = self.encoder(x)
        _, _, z = self.sampling(encoded)
        decoded = self.decoder(z)
        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        mean, _, _ = self.sampling(encoded)
        return mean


class CAE(tf.keras.Model):
    def __init__(self, filters_base=8, pooling=2, k_size=3, n_conv=2, strides=1,
                 activation='selu', dropout=None, latent_dim=0.2, regularizer=None, output_activation='sigmoid'):

        super(CAE, self).__init__()
        self.filters_base = filters_base
        self.pooling = pooling
        self.n_conv = n_conv
        self.strides = strides
        self.activation = activation
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.regularizer = regularizer
        self.output_activation = output_activation
        self.encoder = None
        self.decoder = None

        if isinstance(k_size, (int, np.int)):
            self.k_size = np.repeat(k_size, self.n_conv)
        else:
            self.k_size = k_size

    def build(self, input_shape):
        input_shape = input_shape[1:]
        if isinstance(self.latent_dim, float):
            self.latent_dim = np.round(self.latent_dim * np.prod(input_shape)).astype(int)
        self.get_encoder_decoder(input_shape)

    def get_encoder_decoder(self, input_shape, latent_activation='sigmoid'):
        layers_encoder = [layers.Input(input_shape)]
        layers_decoder = []
        if self.dropout and self.dropout > 0:
            layers_encoder.append(layers.Dropout(self.dropout))
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
        self.encoder = tf.keras.Sequential(layers_encoder, name='Encoder')
        decoder_output = encoder_output = self.encoder.compute_output_shape((None,) + input_shape)
        self.encoder.add(layers.Flatten())
        self.encoder.add(
            layers.Dense(self.latent_dim, activation=latent_activation, activity_regularizer=self.regularizer))
        for layer in layers_decoder:
            decoder_output = layer.compute_output_shape(decoder_output)
        layers_decoder[0:0] = [layers.Dense(np.prod(encoder_output[1:]), activation=self.activation),
                               layers.Reshape(encoder_output[1:])]
        layers_decoder.append(
            layers.Conv2DTranspose(input_shape[-1], input_shape[1] + 1 - decoder_output[1], activation='sigmoid'))
        self.decoder = tf.keras.Sequential(layers_decoder, name='Decoder')

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class CVAE(CAE):
    def __init__(self, filters_base=8, pooling=2, k_size=3, n_conv=2, strides=1,
                 activation='selu', latent_dim=0.2, output_activation='sigmoid'):
        super().__init__(filters_base=filters_base,
                         pooling=pooling,
                         k_size=k_size,
                         n_conv=n_conv,
                         strides=strides,
                         activation=activation,
                         latent_dim=latent_dim,
                         output_activation=output_activation)
        self.regularizer = None
        self.sampling = Sampling()

    def build(self, input_shape):
        input_shape = input_shape[1:]
        self.latent_dim *= 2
        self.regularizer = LatentLossRegularizer(1 / np.prod(input_shape))
        if isinstance(self.latent_dim, float):
            self.latent_dim = np.round(self.latent_dim * np.prod(input_shape)).astype(int)
        self.get_encoder_decoder(input_shape, latent_activation='linear')

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        _, _, z = self.sampling(encoded)
        decoded = self.decoder(z)
        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        mean, _, _ = self.sampling(encoded)
        return mean


class CDAE(CAE):
    def __init__(self, filters_base=8, pooling=2, k_size=3, n_conv=2, strides=1,
                 activation='selu', latent_dim=0.2, output_activation='sigmoid'):
        super().__init__(filters_base=filters_base,
                         pooling=pooling,
                         k_size=k_size,
                         n_conv=n_conv,
                         strides=strides,
                         activation=activation,
                         latent_dim=latent_dim,
                         dropout=0.2,
                         output_activation=output_activation)


class CSAE(CAE):
    def __init__(self, filters_base=8, pooling=2, k_size=3, n_conv=2, strides=1,
                 activation='selu', latent_dim=0.2, output_activation='sigmoid'):
        super().__init__(filters_base=filters_base,
                         pooling=pooling,
                         k_size=k_size,
                         n_conv=n_conv,
                         strides=strides,
                         activation=activation,
                         latent_dim=latent_dim,
                         regularizer=KLDRegularizer(weight=0.05),
                         output_activation=output_activation)
