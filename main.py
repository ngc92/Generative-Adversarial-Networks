import dbas
import tensorflow as tf

from gan import ConvNetDiscriminator, DeconvGenerator, GAN

gen = DeconvGenerator(layers=2, filters=32)
dis = ConvNetDiscriminator(layers=2, filters=32)

gan = GAN(gen, dis, tf.train.AdamOptimizer(), tf.train.AdamOptimizer(), 100)

gan = tf.estimator.Estimator(gan.model_fn, "mnist_gan")


def input_fn(data, shuffle):
    return tf.estimator.inputs.numpy_input_fn({"image": data.data}, data.labels, shuffle=shuffle, num_epochs=1,
                                              batch_size=32)


def input_from_files():
    def make_input():
        filenames = tf.train.match_filenames_once("*.jpg")
        filenames = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        _, image_file = reader.read(filenames)
        image = tf.image.decode_jpeg(image_file)
        resized = tf.image.resize_images(image, [64, 64])
        images = tf.train.shuffle_batch([resized], batch_size=32, capacity=100, min_after_dequeue=10)

        return {"image": images}, None


def random_input_fn(size):
    return lambda: {"z": tf.random_uniform((1, size), -1.0, 1.0), "image": tf.zeros((1, 28, 28, 1))}


mnist = dbas.datasets.MNIST()

for i in range(100):
    gan.train(input_fn=input_fn(mnist.train, True))
    print(gan.evaluate(input_fn(mnist.train.subset(200, 300), shuffle=False)))
