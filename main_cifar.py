import dbas
import tensorflow as tf

from gan import ConvNetDiscriminator, DeconvGenerator, GAN

gen = DeconvGenerator(layers=2, filters=32, regularizer=tf.contrib.layers.l2_regularizer(1e-4))
dis = ConvNetDiscriminator(layers=2, filters=32, regularizer=tf.contrib.layers.l2_regularizer(1e-4))

gan = GAN(gen, dis, tf.train.AdamOptimizer(), tf.train.AdamOptimizer(), 100)

gan = tf.estimator.Estimator(gan.model_fn, "cifar_gan")


def input_fn(data, shuffle):
    return tf.estimator.inputs.numpy_input_fn({"image": data.data}, data.labels, shuffle=shuffle, num_epochs=1,
                                              batch_size=64)

cifar = dbas.datasets.CIFAR()

for i in range(100):
    gan.train(input_fn=input_fn(cifar.train, True))
    print(gan.evaluate(input_fn(cifar.train.subset(200, 300), shuffle=False)))
