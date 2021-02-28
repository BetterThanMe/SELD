import tensorflow as tf
from tensorflow.keras.layers import Dense


# GET DATA
input_data = tf.random.normal([100, 20, 10])
label = tf.random.uniform(shape=[100, 2], maxval=2, dtype=tf.int32)

# GET MODEL


