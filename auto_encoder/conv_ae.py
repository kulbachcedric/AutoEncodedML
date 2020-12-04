import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from classifier.utils import DenseTranspose


class ConvAutoencoder(tf.keras.Model):
    def __init__(self, input_shape, filters_base=8, pooling=2, k_size=3, n_conv=2, strides=1,
                 activation='relu', dropout=0, latent_dim=20):

        super(ConvAutoencoder, self).__init__()

        self.filters_base = filters_base
        self.pooling = pooling
        self.n_conv = n_conv
        self.strides = strides
        self.activation = activation
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.layers_encoder = [layers.Input(input_shape)]
        self.layers_decoder = []

        if isinstance(k_size, (int, np.int)):
            self.k_size = np.repeat(k_size, self.n_conv)
        else:
            self.k_size = k_size

        self.layers_encoder.append(layers.Dropout(0.2))

        for i in range(self.n_conv):
            k_size_i = (self.k_size[i], self.k_size[i])
            self.layers_encoder.append(
                layers.Conv2D(self.filters_base * 2 ** i, kernel_size=k_size_i,
                              activation=self.activation, padding='same',
                              strides=self.strides))

            if self.pooling > 0:
                self.layers_encoder.append(layers.MaxPool2D(pool_size=self.pooling))

            self.layers_decoder.append(
                layers.Conv2DTranspose(self.filters_base * 2 ** (self.n_conv - i - 1),
                                       kernel_size=k_size_i, activation=self.activation,
                                       padding='same', strides=self.strides))

            if self.pooling > 0:
                self.layers_decoder.append(layers.UpSampling2D(size=self.pooling))

        self.encoder = tf.keras.Sequential(self.layers_encoder)

        self.encoder_output = self.encoder.compute_output_shape((None,) + input_shape)
        self.encoder.add(layers.Flatten())
        if self.dropout >= 0:
            self.encoder.add(layers.Dropout(self.dropout))
        self.bottleneck = layers.Dense(latent_dim, activation=self.activation)
        self.encoder.add(self.bottleneck)

        self.layers_decoder[0:0] = [DenseTranspose(self.bottleneck, activation=self.activation),
                                    layers.Reshape(self.encoder_output[1:])]
        self.decoder = tf.keras.Sequential(self.layers_decoder)
        decoder_output = self.decoder.compute_output_shape((None,) + (self.latent_dim,))

        self.decoder.add(
            layers.Conv2DTranspose(input_shape[-1], input_shape[1] + 1 - decoder_output[1], activation='sigmoid'))

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
