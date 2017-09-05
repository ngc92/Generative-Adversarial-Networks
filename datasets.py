from itertools import count

import tensorflow as tf
from glob import glob


def input_from_files(pattern):
    def make_input():
        with tf.variable_scope("input_fn"):
            filenames = tf.train.match_filenames_once(pattern)
            filenames = tf.train.string_input_producer(filenames, 1, shuffle=False, capacity=10)
            reader = tf.WholeFileReader()
            file_name, image_file = reader.read(filenames)
            image = tf.image.decode_jpeg(image_file, name="decode")

        return {"image": image_file, "file_name": file_name, "shape": tf.shape(image)}, None
    return make_input


def model_fn(features, labels, mode):
    predictions = {"image": features["image"],
                   "file_name": features["file_name"],
                   "width": features["shape"][0],
                   "height": features["shape"][1],
                   "channels": features["shape"][2]}
    return tf.estimator.EstimatorSpec(mode=mode, loss=tf.zeros(shape=()), train_op=tf.no_op(), eval_metric_ops={},
                                      predictions=predictions)

path = "/home/erik/celebA/*.jpg"


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def images_to_tfrecords(writer: tf.python_io.TFRecordWriter, path: str):
    features, _ = input_from_files(path)()
    spec = model_fn(features, None, tf.estimator.ModeKeys.PREDICT)

    with tf.Session() as sess:

        # initialize the variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for i in count():
                image = sess.run(spec.predictions)
                features = tf.train.Features(
                    feature={
                        "image":     _bytes_feature(image["image"]),
                        "shape":     _int64_feature([image["width"], image["height"], image["channels"]]),
                        "file_name": _bytes_feature(image["file_name"])
                    }
                )
                example = tf.train.Example(features=features)

                writer.write(example.SerializeToString())

                if i % 1000 == 0:
                    print("Processed %i images" % i)
        finally:
            # stop our queue threads and properly close the session
            coord.request_stop()
            coord.join(threads)


with tf.python_io.TFRecordWriter("celebA.tfrecords") as writer:
    images_to_tfrecords(writer, path)
