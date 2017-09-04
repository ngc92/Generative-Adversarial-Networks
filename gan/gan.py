import functools
from collections import namedtuple

import tensorflow as tf
import gan
from gan.tfutil import log_vars



class GAN:
    FakeLoss = namedtuple("FakeLoss", ["generator", "discriminator", "accuracy"])
    RealLoss = namedtuple("RealLoss", ["discriminator", "accuracy"])

    def __init__(self, generator: gan.Generator, discriminator: gan.Discriminator,
                 gen_optimizer: tf.train.Optimizer, dis_optimizer: tf.train.Optimizer,
                 z_size: int):
        self._generator = generator
        self._discriminator = discriminator
        self._generator_optimizer = gen_optimizer
        self._discriminator_optimizer = dis_optimizer
        self._z_size = z_size

    @property
    def generator(self) -> gan.Generator:
        return self._generator

    @property
    def discriminator(self) -> gan.Discriminator:
        return self._discriminator

    def get_fake_loss(self, features, labels, mode):
        rand_z = tf.random_uniform(shape=(tf.shape(features["image"])[0], tf.constant(self._z_size)),
                                   minval=-1.0, maxval=1.0, name="z_sample")
        with tf.variable_scope("generator", reuse=False):
            result = self.generator(rand_z, mode)
            generated = result.image

            tf.summary.image("fake_images", tf.maximum(generated, 0.0))
            tf.summary.histogram("unbounded_pixels", result.unscaled)
            tf.summary.histogram("fake_image_pixels", generated)

        with tf.variable_scope("discriminator", reuse=False):
            p, l = self.discriminator(generated, mode)
            tf.summary.histogram("fake_p", p)
            tf.summary.histogram("fake_l", l)

        # Train the generator
        gen_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(l), logits=l,
                                                   scope="generator_loss", reduction=tf.losses.Reduction.MEAN)
        dis_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(l), logits=l,
                                                   scope="discriminator_fake_loss", reduction=tf.losses.Reduction.MEAN)

        tf.summary.scalar("train/fake_loss", dis_loss)
        tf.summary.scalar("train/generator_loss", gen_loss)
        accuracy = tf.metrics.accuracy(tf.zeros_like(l), tf.round(p), name="fake_accuracy")

        return self.FakeLoss(gen_loss, dis_loss, accuracy)

    def get_real_loss(self, features, labels, mode):
        with tf.variable_scope("discriminator", reuse=True):
            p, l = self.discriminator(features["image"], mode)

            tf.summary.image("real_images", features["image"])
            tf.summary.histogram("real_p", p)
            tf.summary.histogram("real_l", l)

        dis_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(l), logits=l,
                                                   scope="discriminator_real_loss", reduction=tf.losses.Reduction.MEAN)
        tf.summary.scalar("train/real_loss", dis_loss)
        accuracy = tf.metrics.accuracy(tf.ones_like(l), tf.round(p), name="real_accuracy")

        return self.RealLoss(dis_loss, accuracy)

    def generate(self, features, labels):
        with tf.variable_scope("generator", reuse=True):
            generated = self.generator(features["image"], tf.estimator.ModeKeys.PREDICT)
        return generated

    def __call__(self, features, labels, mode):
        self.generator.set_output_shape(features["image"].shape.as_list()[1:])

        fake_loss = self.get_fake_loss(features, labels, mode)
        real_loss = self.get_real_loss(features, labels, mode)

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
            train_op = tf.group(train_discriminator(), train_generator())
        else:
            train_op = None

        if mode == tf.estimator.ModeKeys.PREDICT:
            evals = {}
            predictions = {
                "generated": self.generate(features, labels)
            }
        else:
            with tf.name_scope("metrics"):
                evals = {
                    "discriminator_fake_loss": tf.metrics.mean(fake_loss.discriminator),
                    "discriminator_real_loss": tf.metrics.mean(real_loss.discriminator),
                    "generator_loss": tf.metrics.mean(generator_loss),
                    "discriminator_loss": tf.metrics.mean(discriminator_loss),
                    "accuracy/fake": fake_loss.accuracy,
                    "accuracy/real": real_loss.accuracy
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
