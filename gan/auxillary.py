import abc
import enum
import tensorflow as tf

from gan.tfutil import track_weights, ScopedCall
from .discriminator import DiscriminatorResult


# TODO python 3.6. Flag enums would be nice here
class AuxillaryLossTarget(enum.Enum):
    DISCRIMINATOR = "discriminator"
    GENERATOR = "generator"
    BOTH = "both"


def _add_with_none(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


class AuxillaryResult:
    def __init__(self, generator=None, discriminator=None, generator_weights=None, discriminator_weights=None,
                 predictions=None):
        if generator is not None:
            assert generator.shape == ()

        if discriminator is not None:
            assert discriminator.shape == ()

        self._generator = generator
        self._discriminator = discriminator

        if generator_weights is None:
            generator_weights = []
        if discriminator_weights is None:
            discriminator_weights = []
        if predictions is None:
            predictions = {}

        self._gen_weights = list(generator_weights)
        self._dis_weights = list(discriminator_weights)
        self._predictions = dict(predictions)

    def __add__(self, other: "AuxillaryResult"):
        predictions = {**self._predictions, **other._predictions}
        return AuxillaryResult(generator=_add_with_none(self._generator, other._generator),
                               discriminator=_add_with_none(self._discriminator, other._discriminator),
                               generator_weights=self._gen_weights + other._gen_weights,
                               discriminator_weights=self._dis_weights + other._dis_weights,
                               predictions=predictions
                               )

    @property
    def generator(self):
        if self._generator is not None:
            return self._generator
        else:
            return tf.zeros((), dtype=tf.float32, name="no_auxillary_loss")

    @property
    def discriminator(self):
        if self._discriminator is not None:
            return self._discriminator
        else:
            return tf.zeros((), dtype=tf.float32, name="no_auxillary_loss")

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
class Auxillary(abc.ABC):
    def __init__(self, target: AuxillaryLossTarget):
        super(Auxillary, self).__init__()
        self._target = target
        self._strength = 1.0

    @property
    def target(self):
        return self._target

    @track_weights
    def __call__(self, discriminator_real: DiscriminatorResult, discriminator_fake: DiscriminatorResult,
                 features: dict, labels: tf.Tensor):
        loss = self._get_loss(discriminator_real, discriminator_fake, features, labels)
        if isinstance(loss, AuxillaryResult):
            return loss
        if self.target == AuxillaryLossTarget.DISCRIMINATOR:
            return AuxillaryResult(discriminator=self._strength*loss, generator=None)
        elif self.target == AuxillaryLossTarget.GENERATOR:
            return AuxillaryResult(discriminator=None, generator=self._strength*loss)
        elif self.target == AuxillaryLossTarget.BOTH:
            return AuxillaryResult(discriminator=self._strength*loss[0], generator=self._strength*loss[1])
        else:
            raise ValueError()

    def set_strength(self, strength):
        self._strength = strength

    @abc.abstractmethod
    def _get_loss(self, d_real: DiscriminatorResult, d_fake: DiscriminatorResult, features: dict, labels: tf.Tensor):
        raise NotImplementedError()


class FeatureMatching(Auxillary):
    """
    Feature matching as described in http://arxiv.org/abs/1606.03498 to improve training.
    """
    def __init__(self):
        super(FeatureMatching, self).__init__(AuxillaryLossTarget.GENERATOR)

    def _get_loss(self, d_real: DiscriminatorResult, d_fake: DiscriminatorResult, features: dict, labels: tf.Tensor):
        # reduction over the mini-batch to estimate E[f(x)], x ~ p_data, E[f(G(z))],z ~ p_sample
        mean_real_features = tf.reduce_mean(d_real.features, axis=0)
        mean_fake_features = tf.reduce_mean(d_fake.features, axis=0)
        discrepancy = tf.losses.mean_squared_error(mean_real_features, mean_fake_features,
                                                   scope="feature_discrepancy", reduction=tf.losses.Reduction.MEAN)
        return discrepancy


class AuxillaryClassifier(Auxillary):
    def __init__(self, num_classes):
        super(AuxillaryClassifier, self).__init__(AuxillaryLossTarget.BOTH)
        self._classes = num_classes

    def _get_loss(self, d_real: DiscriminatorResult, d_fake: DiscriminatorResult, features: dict, labels: tf.Tensor):
        onehot_labels = tf.one_hot(labels, self._classes, dtype=tf.float32)
        real_logits = tf.layers.dense(d_real.features, self._classes, name="logits", reuse=False)
        real_loss = tf.losses.softmax_cross_entropy(onehot_labels, real_logits, scope="real_loss",
                                                    reduction=tf.losses.Reduction.MEAN)

        predicted_label = tf.argmax(real_logits, axis=1)
        real_loss = tf.Print(real_loss, [tf.reduce_mean(tf.cast(tf.equal(tf.argmax(real_logits, axis=1), labels),
                                                                tf.float32))])

        fake_logits = tf.layers.dense(d_fake.features, self._classes, name="logits", reuse=True)
        fake_loss = tf.losses.softmax_cross_entropy(onehot_labels, fake_logits, scope="fake_loss",
                                                    reduction=tf.losses.Reduction.MEAN)

        with tf.name_scope("accuracy"):
            real_logits = tf.Print(real_logits, [tf.argmax(real_logits, axis=1), labels], summarize=10)
            accuracy = tf.summary.scalar("classify_accuracy",
                                         tf.reduce_mean(tf.cast(tf.equal(tf.argmax(real_logits, axis=1), labels),
                                                                tf.float32)))

        return AuxillaryResult(discriminator=real_loss, generator=fake_loss, generator_weights=[],
                               predictions={"label": predicted_label})
