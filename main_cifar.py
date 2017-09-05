import dbas
import tensorflow as tf

from gan import ConvNetDiscriminator, DeconvGenerator, GAN, ClassConditioner
import numpy as np

gen = DeconvGenerator(layers=2, filters=32, regularizer=tf.contrib.layers.l2_regularizer(1e-4))
dis = ConvNetDiscriminator(layers=2, filters=32, regularizer=tf.contrib.layers.l2_regularizer(1e-4), dropout=0.5)

gan = GAN(gen, dis, 100)
gan.add_conditioning(ClassConditioner(10))

gan = tf.estimator.Estimator(gan.model_fn, "cifar_gan")


def input_fn(data, shuffle):
    return tf.estimator.inputs.numpy_input_fn({"image": data.data}, np.argmax(data.labels, axis=1),
                                              shuffle=shuffle, num_epochs=1, batch_size=64)


def random_input_fn(size, classes):
    return lambda: {"latent": tf.random_uniform((len(classes), size), -1.0, 1.0),
                    "labels": tf.constant(classes, dtype=tf.int64),
                    "image": tf.zeros((len(classes), 32, 32, 3))}

cifar = dbas.datasets.CIFAR()

for i in range(100):
    gan.train(input_fn=input_fn(cifar.train, True))
    print(gan.evaluate(input_fn(cifar.test, shuffle=False)))

    predictor = gan.predict(random_input_fn(100, np.arange(0, 10, 1)), predict_keys=["generated"])
    i = 0
    for prediction in predictor:
        import scipy.misc

        scipy.misc.imsave("%i.png" % i, prediction["generated"])
        i += 1
        if i == 10:
            break

