import tensorflow as tf


def input_images_from_tfrecords(file_pattern, batch_size=32, num_epochs=1, crop_size=108, image_size=64):
    def make_input():
        with tf.variable_scope("input_fn"):
            reader = tf.TFRecordReader()
            filename_queue = tf.train.string_input_producer([file_pattern], num_epochs=num_epochs)
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image': tf.FixedLenFeature([], tf.string),
                    'shape': tf.FixedLenFeature((3,), tf.int64),
                })
            image = tf.image.decode_jpeg(features["image"])
            resized = preprocess_images(image, crop_size, image_size)

            # shuffling and batching
            images = tf.train.shuffle_batch([resized], batch_size=batch_size, capacity=200, min_after_dequeue=10,
                                            num_threads=2)

        return {"image": images}, None
    return make_input


def preprocess_images(image, crop_size=108, image_size=64):
    # cropping and resizing
    cropped = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    resized = tf.image.resize_images(cropped, [image_size, image_size])
    resized.set_shape([image_size, image_size, 3])

    # transform to float in [-1, 1]
    resized = (tf.cast(resized, tf.float32) / 255.0) * 2.0 - 1.0
    return resized