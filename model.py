import tensorflow as tf
import math
import numpy as np


def make_discriminator(num_layers, filters):
    def discriminator_fn(features, labels, mode):
        image = features["image"]

        h = image
        for layer in range(num_layers):
            h = tf.layers.conv2d(h, filters, kernel_size=5, strides=(2, 2), padding="same", name="d_h%i_conv" % layer,
                                 activation=tf.nn.relu)
            #h = tf.layers.batch_normalization(h, training=(mode == tf.estimator.ModeKeys.TRAIN))

        flat_size = np.prod(h.shape.as_list()[1:])
        flat = tf.reshape(h, shape=(tf.shape(h)[0], flat_size))
        h4 = tf.layers.dense(flat, 1, name="d_h%i_lin" % (layer+1))

        return tf.nn.sigmoid(h4), h4

    return discriminator_fn


def crop_to_fit(data, width: int, height: int):
    w, h = tuple(data.shape.as_list()[1:3])
    w_err = w - width
    h_err = h - height
    if w_err == 0 and h_err == 0:
        return data
    xl, yl = int(w_err / 2), int(h_err / 2)
    xh, yh = xl + width, yl + height
    return data[:, xl:xh, yl:yh, :]


def make_generator(width, height, channels, num_layers):
    def generator_fn(features, labels, mode):
        s_h, s_w = height, width
        s_sh, s_sw = [s_h], [s_w]

        filters = channels * 2**num_layers
        for i in range(num_layers):
            s_sh += [int(math.ceil(float(s_sh[-1]) / 2))]
            s_sw += [int(math.ceil(float(s_sw[-1]) / 2))]

        # generate initial data
        z = tf.layers.dense(features["z"], s_sh[-1]*s_sw[-1]*filters, activation=tf.nn.relu)
        h = tf.reshape(z, (-1, s_sw[-1], s_sh[-1], filters))

        for layer in range(num_layers):
            filters = int(filters / 2)
            h = tf.layers.conv2d_transpose(h, filters, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)
            #h = tf.layers.batch_normalization(h, training=(mode == tf.estimator.ModeKeys.TRAIN))
            h = crop_to_fit(h, s_sw[-2-layer], s_sh[-2-layer])

        assert tuple(h.shape.as_list()[1:]) == (s_w, s_h, channels), h.shape.as_list()[1:]

        return tf.nn.tanh(h)

    return generator_fn


def make_GAN(generator, discriminator, gen_optimizer: tf.train.Optimizer, dis_optimizer: tf.train.Optimizer, z_size):
    def train_on_fake(features, labels, mode):
        with tf.variable_scope("generator", reuse=False):
            rand_z = tf.random_uniform(shape=(tf.shape(features["image"])[0], tf.constant(z_size)),
                                       minval=0.0, maxval=1.0)
            generated = generator({"z": rand_z}, None, mode)

            tf.summary.image("fake_images", generated)

        generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")

        with tf.variable_scope("discriminator", reuse=False):
            p, l = discriminator({"image": generated}, None, mode)
            tf.summary.histogram("fake_p", p)
            tf.summary.histogram("fake_l", l)

        discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

        # Train the generator
        gen_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(l), logits=l,
                                                   scope="generator_loss", reduction=tf.losses.Reduction.MEAN)
        dis_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(l), logits=l,
                                                   scope="discriminator_fake_loss", reduction=tf.losses.Reduction.MEAN)

        train_gen = gen_optimizer.minimize(gen_loss, None, var_list=generator_vars,
                                           name="generator_training")
        with tf.control_dependencies([train_gen]):
            train_gen = gen_optimizer.minimize(gen_loss, None, var_list=generator_vars,
                                               name="generator_training")

        def train_discriminator():
            return dis_optimizer.minimize(dis_loss, var_list=discriminator_vars,
                                          name="discriminator_fake_training")

        train_dis = tf.cond(dis_loss > 0.1, train_discriminator, lambda: tf.no_op())

        tf.summary.scalar("dis_fake_loss", dis_loss)
        fake_acc = tf.metrics.accuracy(tf.zeros_like(l), tf.round(p), name="fake_accuracy")

        train_op = tf.group(train_gen, train_dis)

        return gen_loss, dis_loss, train_op, fake_acc

    def train_on_real(features, labels, mode):
        with tf.variable_scope("discriminator", reuse=True):
            p, l = discriminator(features, None, mode)

            tf.summary.image("real_images", features["image"])
            tf.summary.histogram("real_p", p)
            tf.summary.histogram("real_l", l)

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        dis_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(l), logits=l,
                                                   scope="discriminator_real_loss", reduction=tf.losses.Reduction.MEAN)
        tf.summary.scalar("dis_real_loss", dis_loss)

        def train():
            return dis_optimizer.minimize(dis_loss, tf.train.get_global_step(), var_list=vars,
                                          name="discriminator_real_training")

        train_disc = tf.cond(dis_loss > 0.1, train, lambda: tf.no_op())

        real_acc = tf.metrics.accuracy(tf.ones_like(l), tf.round(p), name="real_accuracy")
        real_p = tf.metrics.mean(p, name="real_p")

        return dis_loss, train_disc, real_acc, real_p

    def generate(features, labels):
        with tf.variable_scope("generator", reuse=True):
            generated = generator(features, None, tf.estimator.ModeKeys.PREDICT)
        return generated

    def model_fn(features, labels, mode):
        gen_loss, dis_f_loss, train_f_op, facc = train_on_fake(features, labels, mode)
        dis_r_loss, train_r_op, racc, rp = train_on_real(features, labels, mode)

        loss = dis_f_loss + gen_loss + dis_r_loss

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.group(train_f_op, train_r_op)
        else:
            train_op = None

        if mode == tf.estimator.ModeKeys.PREDICT:
            evals = {}
            predictions = {
                "generated": generate(features, labels)
            }
        else:
            evals = {
                "discriminator_fake_loss": tf.metrics.mean(dis_f_loss),
                "discriminator_real_loss": tf.metrics.mean(dis_r_loss),
                "generator_loss": tf.metrics.mean(gen_loss),
                "fake_accuracy": facc,
                "real_accuracy": racc,
                "real_p": rp
            }
            predictions = {}

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=evals,
                                          predictions=predictions)

    return model_fn
