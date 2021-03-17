import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter

from auto_encoder.component import DenseTranspose, Sampling, LatentLossRegularizer, KLDRegularizer


class AE(tf.keras.Model):
    def __init__(self, hidden_dims=0.35, n_layers=3, activation='selu', tied_weights=True, dropout=None,
                 noise_corruption=None, hidden_dropout=None, regularizer=None, output_activation='sigmoid',
                 latent_activation='sigmoid'):

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
        self.noise_corruption = noise_corruption
        self.latent_dim = None
        self.latent_activation = latent_activation
        self.hidden_dropout = hidden_dropout

    def build(self, input_shape):
        input_shape = input_shape[1:]
        self.hidden_dims = self.calculate_hidden_dims(hidden_dims=self.hidden_dims, n_layers=self.n_layers,
                                                      input_shape=input_shape)
        self.get_encoder_decoder(input_shape)

    def get_encoder_decoder(self, input_shape, latent_init='glorot_uniform'):
        layers_encoder = []
        layers_decoder = []
        if self.noise_corruption and self.noise_corruption > 0:
            layers_encoder.append(layers.GaussianNoise(self.noise_corruption))

        if self.dropout and self.dropout > 0:
            layers_encoder.append(layers.Dropout(self.dropout))

        for dim in self.hidden_dims[:-1]:
            layers_encoder.append(layers.Dense(dim, activation=self.activation))
            if self.hidden_dropout:
                layers_encoder.append(layers.Dropout(self.hidden_dropout))

        layers_encoder.append(
            layers.Dense(self.hidden_dims[-1], self.latent_activation, kernel_initializer=latent_init,
                         activity_regularizer=self.regularizer))

        if self.tied_weights:
            dense_layers = [layer for layer in layers_encoder[::-1] if isinstance(layer, layers.Dense)]
            for layer in dense_layers[:-1]:
                layers_decoder.append(DenseTranspose(layer, activation=self.activation))
            layers_decoder.append(DenseTranspose(dense_layers[-1], activation=self.output_activation))
        else:
            for dim in self.hidden_dims[-2::-1]:
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

    def decode(self, x):
        return self.decoder(x)


contr_loss_metric = tf.keras.metrics.Mean('contr_loss')


class RAE(AE):
    def __init__(self, hidden_dims=0.35, n_layers=3, activation='tanh', tied_weights=True, dropout=None,
                 noise_corruption=None,
                 output_activation='sigmoid', latent_activation='sigmoid', beta=0.1):
        super(RAE, self).__init__(hidden_dims, n_layers=n_layers, activation=activation, tied_weights=tied_weights,
                                  dropout=dropout, noise_corruption=noise_corruption,
                                  output_activation=output_activation,
                                  latent_activation=latent_activation)
        self.beta = beta


    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with backprop.GradientTape() as tape:
            contractive_loss, y_pred = self.compute_contractive_loss(x)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses) + tf.cast(contractive_loss,
                                                                                       dtype=tf.float32)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        contr_loss_metric.update_state(contractive_loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        output = {m.name: m.result() for m in self.metrics}
        output['contractive_loss'] = contr_loss_metric.result()
        output['loss'] += output['contractive_loss']
        if 'val_loss' in output:
            output['val_loss'] += output['contractive_loss']
        return output

    @tf.function
    def compute_contractive_loss(self, x):
        if self.encoder:
            with backprop.GradientTape() as regularization_tape:
                regularization_tape.watch(x)
                encoded = self.encoder(x, training=True)
            dy_dx = regularization_tape.batch_jacobian(encoded, x)
            try:
                contractive_loss = tf.norm(tf.reduce_mean(dy_dx, axis=0)) ** 2 * self.beta
            except:
                contractive_loss = 0
                print('Regularization loss infeasible.')
            y_pred = self.decoder(encoded, training=True)
        else:
            print('Encoder is None')
            y_pred = self(x, training=True)
            contractive_loss = 0
        return contractive_loss, y_pred


class DAE(AE):
    def __init__(self, hidden_dims=0.35, n_layers=3, activation='selu', output_activation='sigmoid',
                 latent_activation='sigmoid'):
        super(DAE, self).__init__(hidden_dims, n_layers=n_layers, activation=activation, dropout=0.2,
                                  output_activation=output_activation, latent_activation=latent_activation)


class GDAE(AE):
    def __init__(self, hidden_dims=0.35, n_layers=3, activation='selu', output_activation='sigmoid',
                 noise_corruption=0.5,
                 latent_activation='sigmoid'):
        super(GDAE, self).__init__(hidden_dims, n_layers=n_layers, activation=activation,
                                   noise_corruption=noise_corruption, output_activation=output_activation,
                                   latent_activation=latent_activation)


class SAE(AE):
    def __init__(self, hidden_dims=0.35, n_layers=3, activation='selu', output_activation='sigmoid',
                 latent_activation='sigmoid', target=0.1, weight=0.05):
        super(SAE, self).__init__(hidden_dims, n_layers=n_layers, activation=activation, tied_weights=False,
                                  regularizer=KLDRegularizer(weight, target=target),
                                  output_activation=output_activation,
                                  latent_activation=latent_activation)


class VAE(AE):
    def __init__(self, hidden_dims=0.35, n_layers=3, activation='selu', output_activation='sigmoid',
                 latent_activation='linear', beta=None):
        super().__init__(hidden_dims, n_layers=n_layers, activation=activation, tied_weights=False, dropout=None,
                         output_activation=output_activation, latent_activation='linear')
        self.sampling = Sampling()
        self.beta = beta

    def build(self, input_shape):
        input_shape = input_shape[1:]
        if not self.beta:
            self.beta = 1 / np.prod(input_shape)
        self.regularizer = LatentLossRegularizer(self.beta)
        self.hidden_dims = self.calculate_hidden_dims(hidden_dims=self.hidden_dims, n_layers=self.n_layers,
                                                      input_shape=input_shape)
        self.hidden_dims[-1] *= 2
        self.get_encoder_decoder(input_shape, latent_init='zeros')

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
    def __init__(self, filters_base=12, pooling=None, k_size=(7, 5, 3), n_conv=3, strides=2, activation='selu',
                 dropout=None,
                 latent_dim=0.35, regularizer=None, output_activation='sigmoid'):

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

    def get_encoder_decoder(self, input_shape, latent_activation='sigmoid', latent_init='glorot_uniform'):
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

            if self.pooling and self.pooling > 0:
                layers_encoder.append(layers.MaxPool2D(pool_size=self.pooling))

        for i in range(self.n_conv - 1):
            k_size_i = (self.k_size[i], self.k_size[i])
            layers_decoder.append(
                layers.Conv2DTranspose(self.filters_base * 2 ** (self.n_conv - i - 1),
                                       kernel_size=k_size_i, activation=self.activation,
                                       padding='same', strides=self.strides))
            if self.pooling and self.pooling > 0:
                layers_decoder.append(layers.UpSampling2D(size=self.pooling))

        self.encoder = tf.keras.Sequential(layers_encoder, name='Encoder')
        decoder_output = encoder_output = self.encoder.compute_output_shape((None,) + input_shape)
        self.encoder.add(layers.Flatten())
        self.encoder.add(
            layers.Dense(self.latent_dim, activation=latent_activation, activity_regularizer=self.regularizer,
                         kernel_initializer=latent_init))
        for layer in layers_decoder:
            decoder_output = layer.compute_output_shape(decoder_output)
        layers_decoder[0:0] = [layers.Dense(np.prod(encoder_output[1:]), activation=self.activation),
                               layers.Reshape(encoder_output[1:])]
        layers_decoder.append(
            layers.Conv2DTranspose(input_shape[-1], input_shape[1] + 1 - decoder_output[1],
                                   activation=self.output_activation))
        self.decoder = tf.keras.Sequential(layers_decoder, name='Decoder')

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class CVAE(CAE):
    def __init__(self, filters_base=12, pooling=None, k_size=(7, 5, 3), n_conv=3, strides=2, activation='selu',
                 latent_dim=0.35, output_activation='sigmoid', beta=None):
        super().__init__(filters_base=filters_base, pooling=pooling, k_size=k_size, n_conv=n_conv, strides=strides,
                         activation=activation, latent_dim=latent_dim, output_activation=output_activation)
        self.regularizer = None
        self.sampling = Sampling()
        self.beta = beta

    def build(self, input_shape):
        input_shape = input_shape[1:]
        if not self.beta:
            self.beta = 1 / np.prod(input_shape)
        self.regularizer = LatentLossRegularizer(self.beta)
        if isinstance(self.latent_dim, float):
            self.latent_dim = round(self.latent_dim * np.prod(input_shape))
        self.latent_dim *= 2
        self.get_encoder_decoder(input_shape, latent_activation='linear', latent_init='zeros')

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
    def __init__(self, filters_base=12, pooling=None, k_size=(7, 5, 3), n_conv=3, strides=2, activation='selu',
                 latent_dim=0.35, output_activation='sigmoid'):
        super().__init__(filters_base=filters_base, pooling=pooling, k_size=k_size, n_conv=n_conv, strides=strides,
                         activation=activation, dropout=0.2, latent_dim=latent_dim, output_activation=output_activation)


class CSAE(CAE):
    def __init__(self, filters_base=8, pooling=None, k_size=(7, 5, 3), n_conv=3, strides=2, activation='selu',
                 latent_dim=0.35, output_activation='sigmoid'):
        super().__init__(filters_base=filters_base, pooling=pooling, k_size=k_size, n_conv=n_conv, strides=strides,
                         activation=activation, latent_dim=latent_dim, regularizer=KLDRegularizer(weight=0.05),
                         output_activation=output_activation)
