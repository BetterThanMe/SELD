import tensorflow as tf
from ...utils.utils import merge_two_last_dims


class Merge2LastDims(tf.keras.layers.Layer):
    def __init__(self, name: str = "merge_two_last_dims", **kwargs):
        super(Merge2LastDims, self).__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        return merge_two_last_dims(inputs)

    def get_config(self):
        config = super(Merge2LastDims, self).get_config()
        return config