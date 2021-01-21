from tensorflow.keras import layers, losses
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend


class Sampling(layers.Layer):
    def call(self, inputs, **kwargs):
        mean, log_var = tf.split(inputs, 2, axis=-1)
        z = backend.random_normal(tf.shape(log_var)) * backend.exp(log_var / 2) + mean
        return mean, log_var, z


class LatentLossRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, weight=None):
        self.weight = weight

    def __call__(self, inputs, **kwargs):
        if not self.weight:
            self.weight = 1. / inputs.shape[-1]
        mean, logvar = tf.split(inputs, 2, axis=-1)
        latent_loss = -0.5 * backend.sum(1 + logvar - backend.exp(logvar) - backend.square(mean), axis=-1)
        return backend.sum(latent_loss) * self.weight


class DenseTranspose(layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = tf.keras.activations.get(activation)
        super(DenseTranspose, self).__init__(**kwargs)

    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", initializer="zeros",
                                      shape=[self.dense.input_shape[-1]])
        super().build(batch_input_shape)

    def call(self, inputs, **kwargs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)


class KLDRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target

    def __call__(self, inputs):
        mean_activities = tf.keras.backend.mean(inputs, axis=0)
        return self.weight * (losses.kld(self.target, mean_activities)
                              + losses.kld(1. - self.target, 1. - mean_activities))


class CovRegularizer(layers.Layer):
    def __init__(self, weight=1.0):
        super(CovRegularizer, self).__init__(trainable=False)
        self.weight = weight

    def call(self, inputs, **kwargs):
        covariance = tfp.stats.covariance(inputs)

        if covariance.shape[0] <= 1:
            penalty = 0.0
        else:
            penalty = tf.math.reduce_mean(
                tf.math.squared_difference(covariance, tf.math.multiply(covariance, tf.eye(covariance.shape[0]))))

        return self.weight * penalty
