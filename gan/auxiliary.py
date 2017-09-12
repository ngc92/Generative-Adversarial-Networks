import abc
import enum
import tensorflow as tf

from gan.tfutil import track_weights, ScopedCall, CollectedWeights
from .discriminator import DiscriminatorResult


# TODO python 3.6. Flag enums would be nice here
class AuxiliaryLossTarget(enum.Enum):
    DISCRIMINATOR = "discriminator"
    GENERATOR = "generator"
    BOTH = "both"


def _add_with_none(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _mul_with_none(a, b):
    if a is None:
        return None
    if b is None:
        return None
    return a * b


class AuxiliaryResult:
    """
    This class collects all relevant data that results from adding an Auxiliary objective to a GAN graph.
    This includes losses to be added to discriminator and generator, weights added to discriminator and generator,
    and predictions that are produced.
    """
    def __init__(self, generator_loss=None, discriminator_loss=None, generator_weights=None, discriminator_weights=None,
                 predictions=None):
        if generator_loss is not None:
            assert generator_loss.shape == ()

        if discriminator_loss is not None:
            assert discriminator_loss.shape == ()

        self._generator_loss = generator_loss
        self._discriminator_loss = discriminator_loss

        if generator_weights is None:
            generator_weights = ()
        if discriminator_weights is None:
            discriminator_weights = ()
        if predictions is None:
            predictions = {}

        self._gen_weights = set(generator_weights)
        self._dis_weights = set(discriminator_weights)
        self._predictions = dict(predictions)

    def __add__(self, other: "AuxiliaryResult"):
        predictions = {**self._predictions, **other._predictions}
        return AuxiliaryResult(generator_loss=_add_with_none(self._generator_loss, other._generator_loss),
                               discriminator_loss=_add_with_none(self._discriminator_loss, other._discriminator_loss),
                               generator_weights=self._gen_weights | other._gen_weights,
                               discriminator_weights=self._dis_weights | other._dis_weights,
                               predictions=predictions
                               )

    def scale_loss(self, factor):
        self._discriminator_loss = _mul_with_none(self._discriminator_loss, factor)
        self._generator_loss = _mul_with_none(self._generator_loss, factor)

    @property
    def generator_loss(self):
        if self._generator_loss is not None:
            return self._generator_loss
        else:
            return tf.zeros((), dtype=tf.float32, name="no_auxiliary_loss")

    @property
    def discriminator_loss(self):
        if self._discriminator_loss is not None:
            return self._discriminator_loss
        else:
            return tf.zeros((), dtype=tf.float32, name="no_auxiliary_loss")

    @property
    def generator_weights(self):
        return self._gen_weights

    @property
    def discriminator_weights(self):
        return self._dis_weights

    @property
    def predictions(self):
        return self._predictions


@ScopedCall
class Auxiliary(abc.ABC):
    def __init__(self, target: AuxiliaryLossTarget):
        super(Auxiliary, self).__init__()
        self._target = target
        self._strength = 1.0

    @property
    def target(self):
        return self._target

    @track_weights
    def __call__(self, discriminator_real: DiscriminatorResult, discriminator_fake: DiscriminatorResult,
                 features: dict, labels: tf.Tensor):
        loss = self._build(discriminator_real, discriminator_fake, features, labels)
        if not isinstance(loss, AuxiliaryResult):
            raise TypeError("Auxiliary.build should return AuxiliaryResult, got {}".format(loss))

        if loss._discriminator_loss is not None and self.target is AuxiliaryLossTarget.GENERATOR:
            raise ValueError("Specified auxiliary as generator specific but got discriminator loss {}".format(
                loss.discriminator_loss))

        if loss._generator_loss is not None and self.target is AuxiliaryLossTarget.DISCRIMINATOR:
            raise ValueError("Specified auxiliary as discriminator specific but got generator loss {}".format(
                loss.generator_loss))

        loss.scale_loss(self._strength)
        return loss

    def set_strength(self, strength):
        self._strength = strength

    @abc.abstractmethod
    def _build(self, d_real: DiscriminatorResult, d_fake: DiscriminatorResult, features: dict, labels: tf.Tensor) \
            -> AuxiliaryResult:
        raise NotImplementedError()


class FeatureMatching(Auxiliary):
    """
    Feature matching as described in http://arxiv.org/abs/1606.03498 to improve training.
    """
    def __init__(self):
        super(FeatureMatching, self).__init__(AuxiliaryLossTarget.GENERATOR)

    def _build(self, d_real: DiscriminatorResult, d_fake: DiscriminatorResult, features: dict, labels: tf.Tensor):
        # reduction over the mini-batch to estimate E[f(x)], x ~ p_data, E[f(G(z))],z ~ p_sample
        mean_real_features = tf.reduce_mean(d_real.features, axis=0)
        mean_fake_features = tf.reduce_mean(d_fake.features, axis=0)
        discrepancy = tf.losses.mean_squared_error(mean_real_features, mean_fake_features,
                                                   scope="feature_discrepancy", reduction=tf.losses.Reduction.MEAN)

        return AuxiliaryResult(generator_loss=discrepancy)


class AuxiliaryClassifier(Auxiliary):
    def __init__(self, num_classes):
        super(AuxiliaryClassifier, self).__init__(AuxiliaryLossTarget.BOTH)
        self._classes = num_classes

    def _build(self, d_real: DiscriminatorResult, d_fake: DiscriminatorResult, features: dict, labels: tf.Tensor):
        with CollectedWeights() as weights:
            onehot_labels = tf.one_hot(labels, self._classes, dtype=tf.float32)
            real_logits = tf.layers.dense(d_real.features, self._classes, name="logits", reuse=False)
            real_loss = tf.losses.softmax_cross_entropy(onehot_labels, real_logits, scope="real_loss",
                                                        reduction=tf.losses.Reduction.MEAN)

            predicted_label = tf.argmax(real_logits, axis=1)
            fake_logits = tf.layers.dense(d_fake.features, self._classes, name="logits", reuse=True)
            fake_loss = tf.losses.softmax_cross_entropy(onehot_labels, fake_logits, scope="fake_loss",
                                                        reduction=tf.losses.Reduction.MEAN)

            with tf.name_scope("accuracy"):
                accuracy = tf.summary.scalar("classify_accuracy",
                                             tf.reduce_mean(tf.cast(tf.equal(tf.argmax(real_logits, axis=1), labels),
                                                                    tf.float32)))
            with tf.name_scope("classes"):
                tf.summary.histogram("real", labels)
                tf.summary.histogram("fake", tf.argmax(fake_logits, axis=1))

        return AuxiliaryResult(discriminator_loss=real_loss, generator_loss=fake_loss, generator_weights=[],
                               discriminator_weights=weights.get(),
                               predictions={"label": predicted_label})
