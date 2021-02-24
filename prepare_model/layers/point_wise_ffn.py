import tensorflow as tf


class PointWiseFFN(tf.keras.layers.Layer):
    def __init__(self,
                 size,
                 output_size,
                 activation="relu",
                 dropout=0.1,
                 name="point_wise_ffn",
                 **kwargs):
        super(PointWiseFFN, self).__init__(name=name, **kwargs)
        self.ffn1 = tf.keras.layers.Dense(units=size, activation=activation)
        self.do1 = tf.keras.layers.Dropout(dropout)
        self.ffn2 = tf.keras.layers.Dense(units=output_size)
        self.do2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ffn1(inputs, training=training)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(PointWiseFFN, self).get_config()
        conf.update(self.ffn1.get_config())
        conf.update(self.do1.get_config())
        conf.update(self.ffn2.get_config())
        conf.update(self.do2.get_config())
        return conf

