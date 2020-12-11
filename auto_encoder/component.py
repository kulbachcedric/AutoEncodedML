from tensorflow.keras import layers, losses
import tensorflow as tf
import tensorflow_probability as tfp


@tf.keras.utils.register_keras_serializable(package='Custom', name='dense_transpose')
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

    def get_config(self):
        return {'dense': self.dense, 'activation': self.activation, 'biases': self.biases}


@tf.keras.utils.register_keras_serializable(package='Custom', name='kld')
class KLDRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target

    def __call__(self, inputs):
        mean_activities = tf.keras.backend.mean(inputs, axis=0)
        return self.weight * (losses.kld(self.target, mean_activities)
                              + losses.kld(1. - self.target, 1. - mean_activities))

    def get_config(self):
        return {'weight': self.weight, 'target': self.target}


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
