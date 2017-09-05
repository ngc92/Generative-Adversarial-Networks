from collections import namedtuple
import tensorflow as tf
import numpy as np
from .tfutil import lrelu, get_kernel_regularizer, track_weights, do_regularized, concat_condition_vector, ScopedCall
from .conditioner import NoConditioner
import abc


DiscriminatorResult = namedtuple("DiscriminatorResult", ["probability", "logits", "features"])


@ScopedCall
class Discriminator(abc.ABC):
    def __init__(self):
        self._conditioner = NoConditioner()

    def add_conditioning(self, conditioner):
        self._conditioner = conditioner

    @track_weights
    def __call__(self, images, attributes, mode):
        result = self._build(images, attributes, mode)
        assert isinstance(result, DiscriminatorResult)
        return result

    @abc.abstractmethod
    def _build(self, images, attributes, mode):
        raise NotImplementedError()


class ConvNetDiscriminator(Discriminator):
    def __init__(self, layers, filters, kernel_size=5, regularizer=None, dropout=None):
        super(ConvNetDiscriminator, self).__init__()
        self._layers = layers
        self._filters = filters
        self._kernel_size = kernel_size
        self._regularizer = regularizer
        self._dropout = dropout

        self._build = do_regularized(regularizer)(self._build)

    def _build(self, images, attributes, mode):
        h = images
        current_filters = self._filters
        layer = 0

        condition_vector = self._conditioner(attributes)

        for layer in range(self._layers):
            h = concat_condition_vector(h, condition_vector)
            h = self._conv_layer(h, current_filters, mode)
            if self._dropout is not None:
                h = tf.layers.dropout(h, rate=self._dropout, training=(mode == tf.estimator.ModeKeys.TRAIN))
            current_filters *= 2

        flat_size = np.prod(h.shape.as_list()[1:])
        flat = tf.reshape(h, shape=(tf.shape(h)[0], flat_size))
        h4 = tf.layers.dense(flat, 1, name="d_h%i_lin" % (layer + 1), activation=None,
                             kernel_regularizer=get_kernel_regularizer())

        return DiscriminatorResult(probability=tf.nn.sigmoid(h4), logits=h4, features=flat)

    def _conv_layer(self, input_, filters, mode):
        h = tf.layers.conv2d(input_, filters, kernel_size=self._kernel_size, strides=(2, 2),
                             padding="same", activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                             kernel_regularizer=get_kernel_regularizer())
        h = tf.layers.batch_normalization(h, training=(mode == tf.estimator.ModeKeys.TRAIN))
        return lrelu(h)
