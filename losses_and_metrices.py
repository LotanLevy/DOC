

import tensorflow as tf
import numpy as np

class compactnessLoss(tf.keras.losses.Loss):
    def __init__(self, name='compactness_loss'):
        super(compactnessLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):

        n_dim = np.shape(y_pred)[0]  # number of features vecs
        k_dim = np.shape(y_pred)[1]  # feature vec dim
        loss = tf.constant(0.0)
        for i in range(0, n_dim):
            mask = np.ones(n_dim)
            mask[i] = 0
            mask = tf.constant(mask)
            others = tf.boolean_mask(y_pred, mask)

            mean_vec = tf.math.reduce_sum(others, axis=0) / float(n_dim - 1)
            diff = tf.math.subtract(y_pred[i], mean_vec) / float(n_dim)
            loss = tf.math.add(tf.math.reduce_sum(tf.math.pow(diff, 2)), loss)
        return loss


