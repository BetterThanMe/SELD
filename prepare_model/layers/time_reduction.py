import tensorflow as tf


class TimeReduction(tf.keras.layers.Layer):
    def __init__(self, factor: int, name: str = "time_reduction", **kwargs):
        super(TimeReduction, self).__init__(name=name, **kwargs)
        self.factor = factor

    def build(self, input_shape):
        batch_size = input_shape[0]
        feat_dim = input_shape[-1]
        self.reshape = tf.keras.layers.Reshape([batch_size, -1, feat_dim * self.factor])

    def call(self, inputs, **kwargs):
        return self.reshape(inputs)

    def get_config(self):
        config = super(TimeReduction, self).get_config()
        config.update({"factor": self.factor})
        return config

    def from_config(self, config):
        return self(**config)