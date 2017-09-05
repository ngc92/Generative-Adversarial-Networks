import tensorflow as tf
from abc import ABC, abstractmethod


class Conditioner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, attributes):
        raise NotImplementedError()

    def sample(self, batch_size):
        return self.make_attributes(*self._sample(batch_size))

    @abstractmethod
    def make_attributes(self, features, labels):
        raise NotImplementedError()

    @abstractmethod
    def _sample(self, batch_size):
        raise NotImplementedError()


class NoConditioner(Conditioner):
    def __init__(self):
        super(NoConditioner, self).__init__()

    def __call__(self, attributes):
        return None

    def _sample(self, batch_size):
        return None, None

    def make_attributes(self, features, labels):
        return None


class ClassConditioner(Conditioner):
    def __init__(self, classes):
        super(ClassConditioner, self).__init__()
        self._classes = classes

    def __call__(self, attributes):
        return attributes["class"]

    def _sample(self, batch_size):
        return None, tf.random_uniform((batch_size,), minval=0, maxval=self._classes, dtype=tf.int64)

    def make_attributes(self, features, labels):
        return {"class": tf.one_hot(labels, self._classes, dtype=tf.float32)}