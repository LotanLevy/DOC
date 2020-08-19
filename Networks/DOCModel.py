from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NNInterface import NNInterface
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dropout, Activation
import os




import tensorflow as tf


class DOCModel(NNInterface):
    def __init__(self, cls_num, input_size):
        super().__init__()
        self.model_state = "Reference"

        self.ref_model = Sequential(name="Reference Network")
        self.tar_model = Sequential(name="Secondary Network")

        self.build(cls_num, input_size)

        self.ref_model.summary()
        self.tar_model.summary()

    def build(self, cls_num, input_size):
        input = tf.keras.layers.Input(shape=(input_size[0], input_size[1], 3), name="input")
        self.ref_model.add(input)
        self.tar_model.add(input)

        vgg_conv = vgg16.VGG16(weights=None,
                               include_top=True,
                               classes=cls_num,
                               input_shape=(input_size[0], input_size[1], 3))
        for layer in vgg_conv.layers[:-3]:
            layer.trainable = False
            self.ref_model.add(layer)
            self.tar_model.add(layer)

        fc1 = vgg_conv.layers[-3]
        fc2 = vgg_conv.layers[-2]
        fc3 = tf.keras.layers.Dense(units=cls_num, name='fc3')
        predictions = tf.keras.layers.Activation('softmax')
        dropout1 = tf.keras.layers.Dropout(0.5, name='dropout1')
        dropout2 = tf.keras.layers.Dropout(0.5, name='dropout2')

        self.ref_model.add(fc1)
        self.ref_model.add(dropout1)
        self.ref_model.add(fc2)
        self.ref_model.add(dropout2)
        self.ref_model.add(fc3)
        self.ref_model.add(predictions)

        self.tar_model.add(fc1)
        self.tar_model.add(dropout1)
        self.tar_model.add(fc2)
        self.tar_model.add(fc3)





    def call(self, x, training=True):
        if self.model_state == "Reference":
            return self.ref_model(x, training=training)
        else:
            return self.tar_model(x, training=training)

    def compute_output_shape(self, input_shape):
        return self.__model.compute_output_shape(input_shape)


    def set_losses_and_metrics(self, losses_and_metrics_dict, loss_lambd):
        self.lambd = loss_lambd
        self.losses_and_metrices = losses_and_metrics_dict
        self.trackers = dict()
        for name in losses_and_metrics_dict.keys():
            self.trackers[name] = tf.keras.metrics.Mean(name=name)

    def get_losses_and_metrics_state(self):
        result = dict()
        for tracker in self.trackers:
            result[tracker] = self.trackers[tracker].result()
        return result


    def train_step(self, data):
        input, labels = list(data)[0]

        input_split = int(labels.shape/2)
        ref_inputs = input[:input_split, :,:,:], ref_labels = labels[:input_split]
        tar_inputs = input[input_split:, :, :, :], tar_labels = labels[input_split:]

        with tf.GradientTape() as tape:
            # Descriptiveness loss
            prediction = self.ref_model(ref_inputs, training=True)
            hot_vec = tf.keras.utils.to_categorical([7, 8], num_classes=1000)
            d_loss = self.losses_and_metrices["d_loss"](hot_vec, prediction)
            self.trackers["d_loss"].update_state(d_loss)
        d_gradients = tape.gradient(d_loss, self.ref_model.trainable_variables)


        with tf.GradientTape() as tape:
            # Compactness loss
            prediction = self.tar_model(tar_inputs, training=False)
            c_loss = self.losses_and_metrices["c_loss"](tar_labels, prediction)
            self.trackers["c_loss"].update_state(c_loss)
        c_gradients = tape.gradient(c_loss, self.tar_model.trainable_variables)

        total_gradient = []
        assert (len(d_gradients) == len(c_gradients))
        for i in range(len(d_gradients)):
            total_gradient.append(d_gradients[i] * (1 - self.lambd) + c_gradients[i] * self.lambd)

        self.optimizer.apply_gradients(zip(total_gradient, self.ref_model.trainable_variables))

        return self.get_losses_and_metrics_state()




    def test_step(self, data):
        input, labels = list(data)[0]

        input_split = int(labels.shape / 2)
        ref_inputs = input[:input_split, :, :, :], ref_labels = labels[:input_split]
        tar_inputs = input[input_split:, :, :, :], ref_labels = labels[input_split:]

        with tf.GradientTape() as tape:
            # Descriptiveness loss
            prediction = self.ref_model(ref_inputs, training=False)
            hot_vec = tf.keras.utils.to_categorical([7,8], num_classes=1000)
            d_loss = self.losses_and_metrices["d_loss"](hot_vec, prediction)
            self.trackers["d_loss"].update_state(d_loss)

        with tf.GradientTape() as tape:
            # Compactness loss
            prediction = self.tar_model(tar_inputs, training=False)
            c_loss = self.losses_and_metrices["c_loss"](tar_labels, prediction)
            self.trackers["c_loss"].update_state(c_loss)


        return self.get_losses_and_metrics_state()