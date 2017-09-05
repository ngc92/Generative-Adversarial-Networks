import tensorflow as tf

from gan import ConvNetDiscriminator, DeconvGenerator, GAN

tf.logging.set_verbosity(tf.logging.INFO)

gen = DeconvGenerator(filters=64, layers=4, regularizer=None)
dis = ConvNetDiscriminator(layers=4, filters=64, dropout=0.5)

gan_fn = GAN(gen, dis, z_size=100)

config = tf.estimator.RunConfig()
config = config.replace(save_summary_steps=100)
gan = tf.estimator.Estimator(gan_fn.model_fn, "celeba_gan", config=config)


def input_from_tfrecords(file):
    def make_input():
        with tf.variable_scope("input_fn"):
            reader = tf.TFRecordReader()
            filename_queue = tf.train.string_input_producer([file], num_epochs=1)
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image': tf.FixedLenFeature([], tf.string),
                    'shape':     tf.FixedLenFeature((3,), tf.int64),
                })
            image = tf.image.decode_jpeg(features["image"])
            # cropping
            cropped = tf.image.resize_image_with_crop_or_pad(image, 108, 108)
            resized = tf.image.resize_images(cropped, [64, 64])
            resized.set_shape([64, 64, 3])
            # transform to float in [-1, 1]
            resized = (tf.cast(resized, tf.float32) / 255.0) * 2.0 - 1.0
            images = tf.train.shuffle_batch([resized], batch_size=32, capacity=200, min_after_dequeue=10, num_threads=2)

        return {"image": images}, None
    return make_input


for i in range(100):
    gan.train(input_fn=input_from_tfrecords("celebA.tfrecords"))
    print(gan.evaluate(input_from_tfrecords("celebA.tfrecords"), steps=1000))
