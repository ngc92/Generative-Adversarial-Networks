from collections import namedtuple
import tensorflow as tf
import math
from .tfutil import get_kernel_regularizer, track_weights, do_regularized, crop_to_fit

GeneratorResult = namedtuple("GeneratorResult", ["image", "unscaled"])


class Generator:
    def __init__(self):
        self._output_width = 0
        self._output_height = 0
        self._output_channels = 0

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
    def __call__(self, latent, mode):
        assert self.output_width > 0 and self.output_height > 0 and self.output_channels > 0
        result = self.make_generator(latent, mode)
        assert isinstance(result, GeneratorResult)
        return result

    def make_generator(self, latent, mode):
        raise NotImplementedError()


class DeconvGenerator(Generator):
    def __init__(self, filters, layers, regularizer=None):
        super(DeconvGenerator, self).__init__()
        self._filters = filters
        self._layers = layers
        self._regularizer = regularizer

        self.make_generator = do_regularized(regularizer)(self.make_generator)

    def make_generator(self, latent, mode):
        s_h, s_w = self.output_width, self.output_height
        s_sh, s_sw = [s_h], [s_w]

        filters = self._filters * 2 ** (self._layers - 1)
        for i in range(self._layers):
            s_sh += [int(math.ceil(float(s_sh[-1]) / 2))]
            s_sw += [int(math.ceil(float(s_sw[-1]) / 2))]

        # generate initial data
        z = latent
        projected = tf.layers.dense(z, s_sh[-1] * s_sw[-1] * filters, name="z_projection", activation=None,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                    kernel_regularizer=get_kernel_regularizer())
        h = tf.layers.batch_normalization(projected, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                          name="batch_norm_projection")
        h = tf.nn.relu(h)
        h = tf.reshape(h, (-1, s_sw[-1], s_sh[-1], filters))

        with tf.variable_scope("z_projection", reuse=True):
            projection_w = tf.get_variable("kernel")
            tf.summary.image("projection_w", projection_w[None, :, :, None])

        for i in range(min(filters, 3)):
            tf.summary.image("z_%i" % i, h[:, :, :, i:1 + i])

        for layer in range(self._layers):
            filters = int(filters / 2)
            if layer == self._layers - 1:
                filters = self.output_channels
            h = tf.layers.conv2d_transpose(h, filters, kernel_size=5, strides=2, padding="same", activation=None,
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           kernel_regularizer=get_kernel_regularizer(), name="deconv_%i" % layer)

            if layer != self._layers - 1:
                h = tf.layers.batch_normalization(h, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                  name="batch_norm_%i" % layer)
                h = tf.nn.relu(h)

            h = crop_to_fit(h, s_sw[-2 - layer], s_sh[-2 - layer])

        assert tuple(h.shape.as_list()[1:]) == (s_w, s_h, self.output_channels), h.shape.as_list()[1:]

        return GeneratorResult(image=tf.nn.tanh(h), unscaled=h)

