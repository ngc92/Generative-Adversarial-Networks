import model
import tensorflow as tf
import dbas
import matplotlib.pyplot as plt

gen = model.make_generator(28, 28, 1, 3)
dis = model.make_discriminator(3, 32)

gan_fn = model.make_GAN(gen, dis, tf.train.AdamOptimizer(1e-4), tf.train.AdamOptimizer(1e-4), 10)

gan = tf.estimator.Estimator(gan_fn, "mnist_gan")


def input_fn(data, shuffle):
    return tf.estimator.inputs.numpy_input_fn({"image": data.data}, data.labels, shuffle=shuffle, num_epochs=1,
                                              batch_size=32)


def random_input_fn(size):
    return lambda: {"z": tf.random_uniform((1, size), 0.0, 1.0), "image": tf.zeros((1, 28, 28, 1))}


mnist = dbas.datasets.MNIST()

for i in range(100):
    gan.train(input_fn=input_fn(mnist.train, True))
    print(gan.evaluate(input_fn(mnist.train.subset(200, 300), shuffle=False)))
