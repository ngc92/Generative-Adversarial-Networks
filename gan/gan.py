import functools
from collections import namedtuple

import tensorflow as tf
import gan
import gan.tfutil as tfutil
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
        self._auxillaries = []

        # caching variables during build process
        self._generator_result = None
        self._d_real_result = None
        self._d_fake_result = None

    def add_conditioning(self, conditioner: gan.Conditioner, generator=True, discriminator=True):
        self._conditioner = conditioner
        if discriminator:
            self._discriminator.add_conditioning(conditioner)
        if generator:
            self._generator.add_conditioning(conditioner)

    def add_auxillary(self, auxillary: gan.Auxiliary, strength=1.0):
        auxillary.set_strength(strength)
        self._auxillaries.append(auxillary)

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
        self._generator_result = result

        # check that we generate images in the format that the discriminator knows from the real
        # examples.
        assert features["image"].dtype == result.image.dtype, "Generator should produce data of same dtype as real " \
                                                              "images "
        with tf.name_scope("generator/"):
            generated = result.image

            tf.summary.image("fake_images", generated)
            tf.summary.histogram("unbounded_pixels", result.unscaled)
            tf.summary.histogram("fake_image_pixels", generated)

        result = self.build_discriminator(generated, conditioning_attributes, mode)
        self._d_fake_result = result
        tf.summary.histogram("discriminator/fake_p", result.probability)
        tf.summary.histogram("discriminator/fake_l", result.logits)

        # Train the generator
        with tf.name_scope(self._loss_scope):

            gen_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(result.logits), logits=result.logits,
                                                       scope="generator_loss",
                                                       reduction=tf.losses.Reduction.MEAN)
            dis_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(result.logits), logits=result.logits,
                                                       scope="discriminator_fake_loss",
                                                       reduction=tf.losses.Reduction.MEAN)

            tf.summary.scalar("fake_loss", dis_loss)
            tf.summary.scalar("generator_loss", gen_loss)
        accuracy = tf.metrics.accuracy(tf.zeros_like(result.logits), tf.round(result.probability), name="fake_accuracy")
        acc_sum = tf.summary.scalar("fake_accuracy",
                                    tfutil.accuracy(tf.zeros_like(result.logits),  tf.round(result.probability)))

        return self.FakeLoss(gen_loss, dis_loss, accuracy)

    def get_real_loss(self, features, labels, mode):
        result = self.build_discriminator(features["image"], self._conditioner.make_attributes(features, labels), mode)
        self._d_real_result = result

        tf.summary.image("discriminator/real_images", features["image"])
        tf.summary.histogram("discriminator/real_p", result.probability)
        tf.summary.histogram("discriminator/real_l", result.logits)
        tf.summary.histogram("discriminator/real_image_pixels", features["image"])

        with tf.name_scope(self._loss_scope):
            # label smoothing. according to [NIPS TUTORIAL and
            # Salimans et al. - 2016 - Improved Techniques for Training GANs, only real samples
            # should be smoothed.
            # see also https://github.com/soumith/ganhacks/issues/10.
            dis_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(result.logits), logits=result.logits,
                                                       label_smoothing=self._label_smoothing,
                                                       scope="discriminator_real_loss",
                                                       reduction=tf.losses.Reduction.MEAN)
            tf.summary.scalar("real_loss", dis_loss)
        accuracy = tf.metrics.accuracy(tf.ones_like(result.logits), tf.round(result.probability), name="real_accuracy")
        acc_sum = tf.summary.scalar("real_accuracy",
                                    tfutil.accuracy(tf.ones_like(result.logits), tf.round(result.probability)))

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

        # build the auxillaries
        aux_loss = gan.AuxiliaryResult()
        for aux in self._auxillaries:
            aux = aux  # type: gan.Auxiliary
            aux_loss += aux(discriminator_real=self._d_real_result, discriminator_fake=self._d_fake_result,
                            features=features, labels=labels,
                            scope="auxiliary")

        def train_discriminator():
            with tf.name_scope(loss_scope):
                regularization_loss = tf.losses.get_regularization_loss("discriminator")
                loss = discriminator_loss + regularization_loss + aux_loss.discriminator_loss
                tf.summary.scalar("auxiliary_disciminator_loss", aux_loss.discriminator_loss)
                tf.summary.scalar("generator_regularizer_loss", regularization_loss)
            print(aux_loss.discriminator_weights)
            vars = discriminator_vars + list(aux_loss.discriminator_weights)
            return self._discriminator_optimizer.minimize(loss, tf.train.get_global_step(), var_list=vars,
                                                          name="discriminator_fake_training")

        def train_generator():
            with tf.name_scope(loss_scope):
                regularization_loss = tf.losses.get_regularization_loss("generator")
                loss = generator_loss + regularization_loss + aux_loss.generator_loss
                tf.summary.scalar("auxiliary_generator_loss", aux_loss.generator_loss)
                tf.summary.scalar("generator_regularizer_loss", regularization_loss)
            vars = generator_vars + list(aux_loss.generator_weights)
            return self._generator_optimizer.minimize(loss, tf.train.get_global_step(), var_list=vars,
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
                "original": features["image"],
#                "class": features["class"],
                "generated": self.generate(features, labels).image
            }
            predictions.update(aux_loss.predictions)

        else:
            with tf.name_scope("metrics"):
                evals = {
                    "metrics/discriminator_fake_loss": tf.metrics.mean(fake_loss.discriminator),
                    "metrics/discriminator_real_loss": tf.metrics.mean(real_loss.discriminator),
                    "metrics/discriminator_aux_loss": tf.metrics.mean(aux_loss.discriminator_loss),
                    "metrics/generator_aux_loss": tf.metrics.mean(aux_loss.generator_loss),
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
