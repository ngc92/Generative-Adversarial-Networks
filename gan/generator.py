from collections import namedtuple
import tensorflow as tf
import math

from .tfutil import get_kernel_regularizer, track_weights, do_regularized, crop_to_fit, concat_condition_vector, \
    ScopedCall
from .conditioner import NoConditioner
import abc

GeneratorResult = namedtuple("GeneratorResult", ["image", "unscaled"])


@ScopedCall
class Generator(abc.ABC):
    def __init__(self):
        self._output_width = 0
        self._output_height = 0
        self._output_channels = 0
        self._conditioner = NoConditioner()

    def add_conditioning(self, conditioner):
        self._conditioner = conditioner

    def set_output_shape(self, shape):
        assert len(shape) == 3
        self._output_width = shape[0]
        self._output_height = shape[1]
        self._output_channels = shape[2]

    @property
    def output_width(self):
        return self._output_width

    @property
    def output_height(self):
        return self._output_height

    @property
    def output_channels(self):
        return self._output_channels

    @track_weights
    def __call__(self, latent, attributes, mode):
        assert self.output_width > 0 and self.output_height > 0 and self.output_channels > 0
        result = self._build(latent, attributes, mode)
        assert isinstance(result, GeneratorResult)
        return result

    @abc.abstractmethod
    def _build(self, latent, attributes, mode):
        raise NotImplementedError()


class DeconvGenerator(Generator):
    def __init__(self, filters, layers, regularizer=None, kernel_size=5, stride=2):
        super(DeconvGenerator, self).__init__()
        self._filters = filters
        self._layers = layers
        self._kernel_size = kernel_size
        self._stride = stride
        self._regularizer = regularizer

        self._build = do_regularized(regularizer)(self._build)

    def _build(self, latent, attributes, mode):
        s_h, s_w = self.output_width, self.output_height
        s_sh, s_sw = [s_h], [s_w]

        filters = self._filters * 2 ** (self._layers - 1)
        for i in range(self._layers):
            s_sh += [int(math.ceil(float(s_sh[-1]) / 2))]
            s_sw += [int(math.ceil(float(s_sw[-1]) / 2))]

        condition_vector = self._conditioner(attributes)

        # generate initial data
        z = latent
        if condition_vector is not None:
            z = tf.concat([z, condition_vector], axis=1)

        projected = tf.layers.dense(z, s_sh[-1] * s_sw[-1] * filters, name="z_projection", activation=None,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                    kernel_regularizer=get_kernel_regularizer())
        # It seems there is no batch-norm here
        #h = tf.layers.batch_normalization(projected, training=(mode == tf.estimator.ModeKeys.TRAIN),
        #                                  name="batch_norm_projection")
        h = projected
        h = tf.nn.relu(h)
        h = tf.reshape(h, (-1, s_sw[-1], s_sh[-1], filters))

        with tf.variable_scope("z_projection", reuse=True):
            projection_w = tf.get_variable("kernel")
            tf.summary.image("projection_w", projection_w[None, :, :, None])

        for i in range(min(filters, 3)):
            tf.summary.image("z_%i" % i, h[:, :, :, i:i+1])

        for layer in range(self._layers):
            filters = int(filters / 2)
            if layer == self._layers - 1:
                filters = self.output_channels

            h = concat_condition_vector(h, condition_vector)

            h = tf.layers.conv2d_transpose(h, filters, kernel_size=self._kernel_size, strides=self._stride,
                                           padding="same", activation=None,
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           kernel_regularizer=get_kernel_regularizer(), name="deconv_%i" % layer)

            # batchnorm and ReLU for all but the last layer.
            if layer != self._layers - 1:
                h = tf.layers.batch_normalization(h, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                  name="batch_norm_%i" % layer)
                h = tf.nn.relu(h)

            h = crop_to_fit(h, s_sw[-2 - layer], s_sh[-2 - layer])

        assert tuple(h.shape.as_list()[1:]) == (s_w, s_h, self.output_channels), h.shape.as_list()[1:]

        return GeneratorResult(image=tf.nn.tanh(h), unscaled=h)

