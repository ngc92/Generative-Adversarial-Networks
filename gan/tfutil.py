import functools
from contextlib import contextmanager

import tensorflow as tf

_TF_LAYERS_KERNEL_REGULARIZER_STACK = []


def get_kernel_regularizer():
    if len(_TF_LAYERS_KERNEL_REGULARIZER_STACK) > 0:
        return _TF_LAYERS_KERNEL_REGULARIZER_STACK[-1]
    else:
        return None


@contextmanager
def do_regularized(regularizer):
    _TF_LAYERS_KERNEL_REGULARIZER_STACK.append(regularizer)
    yield
    _TF_LAYERS_KERNEL_REGULARIZER_STACK.pop()


def lrelu(x):
    return tf.maximum(x, 0.2 * x)


def track_weights(fun):
    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        before = set(tf.trainable_variables())
        result = fun(*args, **kwargs)
        after = set(tf.trainable_variables())
        added = after - before
        with tf.name_scope("weight_summary"):
            for var in added:
                if "batch_norm" not in var.name:
                    name = var.name.split(":")[0]
                    summary = tf.summary.scalar(name, tf.nn.l2_loss(var))
                    print("Found weight %s with shape %s" % (name, var.shape))
            return result
    return wrapped


def crop_to_fit(data, width: int, height: int):
    w, h = tuple(data.shape.as_list()[1:3])
    w_err = w - width
    h_err = h - height
    if w_err == 0 and h_err == 0:
        return data
    xl, yl = int(w_err / 2), int(h_err / 2)
    xh, yh = xl + width, yl + height
    return data[:, xl:xh, yl:yh, :]


def log_vars(vars, message=""):
    tf.logging.info(message)
    for var in vars:
        tf.logging.info("%s", var.name)

