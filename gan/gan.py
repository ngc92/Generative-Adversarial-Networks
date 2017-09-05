import functools
from collections import namedtuple

import tensorflow as tf
import gan
from gan.tfutil import log_vars


class GAN:
    FakeLoss = namedtuple("FakeLoss", ["generator", "discriminator", "accuracy"])
    RealLoss = namedtuple("RealLoss", ["discriminator", "accuracy"])

    def __init__(self, generator: gan.Generator, discriminator: gan.Discriminator,
                 z_size: int, label_smoothing=0.1):
        self._generator = generator
        self._discriminator = discriminator
        self._generator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
        self._discriminator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
        self._z_size = z_size
        self._conditioner = gan.NoConditioner()
        self._label_smoothing = label_smoothing

    def add_conditioning(self, conditioner: gan.Conditioner):
        self._conditioner = conditioner
        self._discriminator.add_conditioning(conditioner)
        self._generator.add_conditioning(conditioner)

    def set_optimizer(self, target, optimizer: tf.train.Optimizer):
        assert isinstance(optimizer, tf.train.Optimizer)
        if target == "both":
            self.set_optimizer("generator", optimizer)
            self.set_optimizer("discriminator", optimizer)
            return

        if target == "generator":
            self._generator_optimizer = optimizer
        elif target == "discriminator":
            self._discriminator_optimizer = optimizer
        else:
            raise ValueError("Invalid target {} specified".format(target))

    def build_generator(self, latent, attributes, mode):
        return self._generator(latent, attributes, mode, scope="generator")

    def build_discriminator(self, image, attributes, mode):
        return self._discriminator(image, attributes, mode, scope="discriminator")

    def sample_latent(self, shape):
        #return tf.random_uniform(shape=shape, minval=-1.0, maxval=1.0, name="z_sample")
        return tf.random_normal(shape=shape, mean=0, stddev=1.0, name="z_sample")

    def get_fake_loss(self, features, labels, mode):
        batch_size = tf.shape(features["image"])[0]
        with tf.name_scope("latent_code"):
            rand_z = self.sample_latent((batch_size, tf.constant(self._z_size)))

        # random conditioning for fakes
        #conditioning_attributes = self._conditioner.sample(batch_size)
        conditioning_attributes = self._conditioner.make_attributes(features, labels)

        result = self.build_generator(rand_z, conditioning_attributes, mode)

        # check that we generate images in the format that the discriminator knows from the real
        # examples.
        assert features["image"].dtype == result.image.dtype, "Generator should produce data of same dtype as real " \
                                                              "images "
        with tf.name_scope("generator/"):
            generated = result.image

            tf.summary.image("fake_images", generated, 0.0)
            tf.summary.histogram("unbounded_pixels", result.unscaled)
            tf.summary.histogram("fake_image_pixels", generated)

        p, l = self.build_discriminator(generated, conditioning_attributes, mode)
        tf.summary.histogram("discriminator/fake_p", p)
        tf.summary.histogram("discriminator/fake_l", l)

        # Train the generator
        with tf.name_scope(self._loss_scope):

            gen_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(l), logits=l,
                                                       scope="generator_loss",
                                                       reduction=tf.losses.Reduction.MEAN)
            dis_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(l), logits=l,
                                                       scope="discriminator_fake_loss",
                                                       reduction=tf.losses.Reduction.MEAN)

            tf.summary.scalar("fake_loss", dis_loss)
            tf.summary.scalar("generator_loss", gen_loss)
        accuracy = tf.metrics.accuracy(tf.zeros_like(l), tf.round(p), name="fake_accuracy")

        return self.FakeLoss(gen_loss, dis_loss, accuracy)

    def get_real_loss(self, features, labels, mode):
        p, l = self.build_discriminator(features["image"], self._conditioner.make_attributes(features, labels), mode)

        tf.summary.image("discriminator/real_images", features["image"])
        tf.summary.histogram("discriminator/real_p", p)
        tf.summary.histogram("discriminator/real_l", l)
        tf.summary.histogram("discriminator/real_image_pixels", features["image"])

        with tf.name_scope(self._loss_scope):
            # label smoothing. according to [NIPS TUTORIAL and
            # Salimans et al. - 2016 - Improved Techniques for Training GANs, only real samples
            # should be smoothed.
            # see also https://github.com/soumith/ganhacks/issues/10.
            dis_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(l), logits=l,
                                                       label_smoothing=self._label_smoothing,
                                                       scope="discriminator_real_loss",
                                                       reduction=tf.losses.Reduction.MEAN)
            tf.summary.scalar("real_loss", dis_loss)
        accuracy = tf.metrics.accuracy(tf.ones_like(l), tf.round(p), name="real_accuracy")

        return self.RealLoss(dis_loss, accuracy)

    def generate(self, features, labels):
        generated = self.build_generator(features["latent"], self._conditioner.make_attributes(features, labels),
                                         tf.estimator.ModeKeys.PREDICT)
        return generated

    def __call__(self, features, labels, mode):
        # set up the scopes
        with tf.name_scope("losses") as loss_scope:
            self._loss_scope = loss_scope

        self._generator.set_output_shape(features["image"].shape.as_list()[1:])

        if mode == tf.estimator.ModeKeys.PREDICT:
            labels = features.get("labels", None)

        fake_loss = self.get_fake_loss(features, labels, mode)
        real_loss = self.get_real_loss(features, labels, mode)

        with tf.name_scope(self._loss_scope):
            generator_loss = fake_loss.generator
            discriminator_loss = fake_loss.discriminator + real_loss.discriminator
            regularization_loss = tf.losses.get_regularization_loss()
            loss = generator_loss + discriminator_loss + regularization_loss

        generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

        log_vars(generator_vars, "Trainable variables of the generator: ")
        log_vars(discriminator_vars, "Trainable variables of the discriminator: ")

        def train_discriminator():
            loss = discriminator_loss + tf.losses.get_regularization_loss("discriminator")
            return self._discriminator_optimizer.minimize(loss,
                                                          tf.train.get_global_step(), var_list=discriminator_vars,
                                                          name="discriminator_fake_training")

        def train_generator():
            return self._generator_optimizer.minimize(generator_loss + tf.losses.get_regularization_loss("generator"),
                                                      tf.train.get_global_step(), var_list=generator_vars,
                                                      name="generator_training")

        if mode == tf.estimator.ModeKeys.TRAIN:
            updates_generator = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "generator")
            updates_discriminator = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "discriminator")
            with tf.control_dependencies(updates_generator+updates_discriminator):
                train_op = tf.group(train_discriminator(), train_generator())
        else:
            train_op = None

        if mode == tf.estimator.ModeKeys.PREDICT:
            evals = {}
            predictions = {
                "generated": self.generate(features, labels).image
            }
        else:
            with tf.name_scope("metrics"):
                evals = {
                    "metrics/discriminator_fake_loss": tf.metrics.mean(fake_loss.discriminator),
                    "metrics/discriminator_real_loss": tf.metrics.mean(real_loss.discriminator),
                    "metrics/generator_loss": tf.metrics.mean(generator_loss),
                    "metrics/discriminator_loss": tf.metrics.mean(discriminator_loss),
                    "metrics/accuracy_fake": fake_loss.accuracy,
                    "metrics/accuracy_real": real_loss.accuracy
                }
            predictions = {}

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=evals,
                                          predictions=predictions)

    """
    It seems that tf has problems with class __call__ functions as model_fn
    """
    @property
    def model_fn(self):
        @functools.wraps(self.__call__)
        def model_fn(features, labels, mode):
            return self(features, labels, mode)
        return model_fn
