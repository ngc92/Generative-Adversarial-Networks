import dbas
import tensorflow as tf
import numpy as np

from gan import ConvNetDiscriminator, DeconvGenerator, GAN, ClassConditioner, FeatureMatching, AuxillaryClassifier

gen = DeconvGenerator(layers=2, filters=32)
dis = ConvNetDiscriminator(layers=2, filters=32)

gan = GAN(gen, dis, 100)
gan.add_conditioning(ClassConditioner(10))
gan.add_auxillary(FeatureMatching())
gan.add_auxillary(AuxillaryClassifier(10))

gan = tf.estimator.Estimator(gan.model_fn, "mnist_gan")


def input_fn(data, shuffle):
    return tf.estimator.inputs.numpy_input_fn({"image": data.data}, np.argmax(data.labels, axis=1).astype(np.int64),
                                              shuffle=shuffle, num_epochs=1, batch_size=32)


def random_input_fn(size, classes):
    return lambda: {"latent": tf.random_uniform((len(classes), size), -1.0, 1.0),
                    "labels": tf.constant(classes, dtype=tf.int64),
                    "image": tf.zeros((len(classes), 28, 28, 1))}


mnist = dbas.datasets.MNIST()

try:
    for i in range(20):
        gan.train(input_fn=input_fn(mnist.train, True))
        print(gan.evaluate(input_fn(mnist.test, shuffle=False)))
except KeyboardInterrupt: pass


predictor = gan.predict(random_input_fn(100, np.arange(0, 10, 1)), predict_keys=["generated"])
i = 0
for prediction in predictor:
    import scipy.misc
    scipy.misc.imsave("%i.png" % i, prediction["generated"][:, :, 0])
    i += 1
    if i == 10:
        break
