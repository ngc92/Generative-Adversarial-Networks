import dbas
import tensorflow as tf

from gan import ConvNetDiscriminator, DeconvGenerator, GAN, ClassConditioner, AuxiliaryClassifier, FeatureMatching
import numpy as np

# as in the AuxGan Paper
gen = DeconvGenerator(layers=3, filters=96, regularizer=tf.contrib.layers.l2_regularizer(1e-4))
dis = ConvNetDiscriminator(layers=6, filters=16, regularizer=tf.contrib.layers.l2_regularizer(1e-4), dropout=0.5,
                           stride=[2, 1, 2, 1, 2, 1], kernel_size=3)

gan = GAN(gen, dis, 100)
gan.add_conditioning(ClassConditioner(10), discriminator=False)
gan.add_auxillary(FeatureMatching())
gan.add_auxillary(AuxiliaryClassifier(10))

gan = tf.estimator.Estimator(gan.model_fn, "cifar_gan")


def input_fn(data, shuffle):
    repeats = None if shuffle else 1
    return tf.estimator.inputs.numpy_input_fn({"image": data.data * 2.0 - 1.0}, np.argmax(data.labels, axis=1),
                                              shuffle=shuffle, num_epochs=repeats, batch_size=64)


def random_input_fn(size, classes):
    return lambda: {"latent": tf.random_uniform((len(classes), size), -1.0, 1.0),
                    "labels": tf.constant(classes, dtype=tf.int64),
                    "image": tf.zeros((len(classes), 32, 32, 3))}

cifar = dbas.datasets.CIFAR()

#gan.train(input_fn=input_fn(cifar.train, True), max_steps=50000*100/64)
#print(gan.evaluate(input_fn(cifar.test, shuffle=False)))

predictor = gan.predict(random_input_fn(100, np.arange(0, 10, 1)), predict_keys=["generated"])
i = 0
j = 0
images = []
row = []
for prediction in predictor:
    row += [prediction["generated"]]

    i += 1
    if i == 10:
        images += [np.concatenate(row, axis=1)]
        row = []
        i = 0
        j += 1
        if j == 10:
            break


import scipy.misc
scipy.misc.imsave("cifar.png", np.concatenate(images, axis=0))