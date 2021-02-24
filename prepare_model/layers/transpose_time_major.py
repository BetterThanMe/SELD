import tensorflow as tf


class TransposeTimeMajor(tf.keras.layers.Layer):
    def __init__(self, name: str = "transpose_time_major", **kwargs):
        super(TransposeTimeMajor, self).__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        return tf.transpose(inputs, perm=[1, 0, 2])

    def get_config(self):
        config = super(TransposeTimeMajor, self).get_config()
        return config