from collections import namedtuple
import tensorflow as tf
import numpy as np
from .tfutil import lrelu, get_kernel_regularizer, track_weights, do_regularized

DiscriminatorResult = namedtuple("DiscriminatorResult", ["probability", "logits"])


class Discriminator:
    def __init__(self):
        pass

    @track_weights
    def __call__(self, images, mode):
        result = self.make_discriminator(images, mode)
        assert isinstance(result, DiscriminatorResult)
        return result

    def make_discriminator(self, images, mode):
        raise NotImplementedError()


class ConvNetDiscriminator(Discriminator):
    def __init__(self, layers, filters, kernel_size=5, regularizer=None):
        super(ConvNetDiscriminator, self).__init__()
        self._layers = layers
        self._filters = filters
        self._kernel_size = kernel_size
        self._regularizer = regularizer

        self.make_discriminator = do_regularized(regularizer)(self.make_discriminator)

    def make_discriminator(self, images, mode):
        h = images
        current_filters = self._filters
        layer = 0
        for layer in range(self._layers):
            h = tf.layers.conv2d(h, current_filters, kernel_size=self._kernel_size, strides=(2, 2),
                                 padding="same", name="d_h%i_conv" % layer,
                                 activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 kernel_regularizer=get_kernel_regularizer())
            h = tf.layers.batch_normalization(h, training=(mode == tf.estimator.ModeKeys.TRAIN))
            h = lrelu(h)
            current_filters *= 2

        flat_size = np.prod(h.shape.as_list()[1:])
        flat = tf.reshape(h, shape=(tf.shape(h)[0], flat_size))
        h4 = tf.layers.dense(flat, 1, name="d_h%i_lin" % (layer + 1), activation=None,
                             kernel_regularizer=get_kernel_regularizer())

        return DiscriminatorResult(probability=tf.nn.sigmoid(h4), logits=h4)
