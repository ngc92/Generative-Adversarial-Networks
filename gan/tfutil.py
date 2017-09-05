import abc
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
        summaries = []
        with tf.name_scope("weight_summary"):
            for var in added:
                if "batch_norm" not in var.name:
                    name = var.name.split(":")[0]
                    summaries.append(("weight_summary/"+name, tf.nn.l2_loss(var)))
                    print("Found weight %s with shape %s" % (name, var.shape))

        # put weight summaries in global scope (they remain unqiue as they contain the full variable name)
        with tf.name_scope(None):
            for n, v in summaries:
                tf.summary.scalar(n, v)
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


def concat_condition_vector(image, condition_vector):
    with tf.name_scope("concat_conditioning_vector"):
        if condition_vector is not None:
            in_shape = image.shape.as_list()
            broadcasted = tf.tile(condition_vector[:, None, None, :], [1, in_shape[1], in_shape[2], 1])
            return tf.concat([image, broadcasted], axis=3)
        else:
            return image


class ScopeInfo:
    def __init__(self):
        self._current_graph = None
        self._reuse = False
        self._scope = None

    @property
    def reuse(self):
        if self._current_graph != tf.get_default_graph():
            self._current_graph = tf.get_default_graph()
            self._reuse = False
        return self._reuse

    @reuse.setter
    def reuse(self, val):
        self._reuse = val

    @property
    def scope(self):
        return self._scope

    @scope.setter
    def scope(self, value):
        assert isinstance(value, tf.VariableScope)
        self._scope = value


def ScopedCall(cls):
    call = cls.__call__
    init = cls.__init__

    @functools.wraps(init)
    def new_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if hasattr(self, "_scope_info"):
            raise AttributeError("Attribute Conflict")
        self._scope_info = ScopeInfo()

    @functools.wraps(call)
    def scoped_call(self, *args, scope, **kwargs):
        reuse = self._scope_info.reuse

        with tf.variable_scope(scope, reuse=reuse) as var_scope:
            print("IN_SCOPE")
            result = call(self, *args, **kwargs)
            self._scope_info.reuse = True
            self._scope_info.scope = var_scope
            return result

    cls.__call__ = scoped_call
    cls.__init__ = new_init
    return cls