import tensorflow as tf

from gan import ConvNetDiscriminator, DeconvGenerator, GAN, FeatureMatching
from gan.input_fn import input_images_from_tfrecords

tf.logging.set_verbosity(tf.logging.INFO)

gen = DeconvGenerator(filters=96, layers=4, regularizer=tf.contrib.layers.l2_regularizer(1e-5))
dis = ConvNetDiscriminator(layers=4, filters=96, regularizer=tf.contrib.layers.l2_regularizer(1e-5), dropout=0.5)

gan_fn = GAN(gen, dis, z_size=100)
gan_fn.add_auxillary(FeatureMatching())

config = tf.estimator.RunConfig()
config = config.replace(save_summary_steps=100)
gan = tf.estimator.Estimator(gan_fn.model_fn, "celeba_gan", config=config)

for i in range(25):
    gan.train(input_fn=input_images_from_tfrecords("celebA.tfrecords"))
    print(gan.evaluate(input_images_from_tfrecords("celebA.tfrecords"), steps=1000))
