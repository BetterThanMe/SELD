import sys
sys.path.append('./prepare_data')
from specaugment import SpecAugment, SpecAugmentTensor
import numpy as np
import tensorflow as tf


def augment_spec(X):
    x_len = len(X)
    if len(X.shape) < 4:
        X = np.expand_dims(X, axis=-1)
    num_channel = X.shape[-1]
    augment = SpecAugment()
    for i in range(x_len):
        for c in range(num_channel):
            X_ = X[i, :, :, c]
            X_ = np.reshape(X_, (-1, X_.shape[0], X_.shape[1], 1))
            freq_masked = augment.freq_mask(X_)  # Applies Frequency Masking to the mel spectrogram
            time_masked = augment.time_mask(freq_masked)  # Applies Time Masking to the mel spectrogram
            X[i, :, :, c] = np.squeeze(time_masked)
    return np.squeeze(X)


def augment_spec_tensor(x):
    x_len = len(x)
    if len(x.shape) < 4:
        x = tf.expand_dims(x, axis=-1)
    num_channel = x.shape[-1]
    augment = SpecAugmentTensor()
    x = tf.keras.backend.variable(x)
    for i in range(x_len):
        for c in range(num_channel):
            x_ = x[i, :, :, c]
            x_ = tf.reshape(x_, (-1, x_.shape[0], x_.shape[1], 1))
            freq_masked = augment.freq_mask(x_)  # Applies Frequency Masking to the mel spectrogram
            time_masked = augment.time_mask(freq_masked)  # Applies Time Masking to the mel spectrogram
            x[i, :, :, c].assign(tf.squeeze(time_masked))
    return tf.squeeze(x)


# a = tf.random.normal(shape=[4, 600, 64, 7], dtype=tf.float32)
# b = augment_spec_tensor(a)
# print(b.shape)


