from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NNInterface import NNInterface
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dropout, Activation
import os
from train_test import Trainer, Validator






import tensorflow as tf


class DOCModel(NNInterface):
    def __init__(self, cls_num, input_size):
        super().__init__()
        self.model_state = "Reference"

        self.ref_model = Sequential(name="reference")
        self.tar_model = Sequential(name="secondary")

        self.build_network(cls_num, input_size)

        self.ref_model.summary()
        self.tar_model.summary()

        self.ready_for_train = False
        self.trainer = None
        self.validator = None



    def build_network(self, cls_num, input_size):
        input = tf.keras.layers.InputLayer(input_shape=(input_size[0], input_size[1], 3), name="input")
        self.ref_model.add(input)
        self.tar_model.add(input)

        vgg_conv = vgg16.VGG16(weights="imagenet",
                               include_top=True,
                               classes=cls_num,
                               input_shape=(input_size[0], input_size[1], 3), classifier_activation=None)
        for layer in vgg_conv.layers[:-3]:
            layer.trainable = False
            self.ref_model.add(layer)
            self.tar_model.add(layer)

        fc1 = vgg_conv.layers[-3]
        fc2 = vgg_conv.layers[-2]
        fc3 = vgg_conv.layers[-1]
        # predictions = vgg_conv.layers[-1]
        # fc3 = tf.keras.layers.Dense(units=cls_num, name='fc3')
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

    def set_ready_for_train(self, optimizer, loss_lambda, losses=dict(), metrics=dict()):
        self.ready_for_train = True
        self.trainer = Trainer("train", losses, metrics, self.ref_model, self.tar_model, loss_lambda, optimizer)
        self.validator = Validator("test", losses, metrics, self.ref_model, self.tar_model)

    def on_validation_epoch_end(self):
        self.validator.reset()





    def train_step(self, ref_inputs, ref_labels, tar_inputs, tar_labels):
        if self.ready_for_train:
            return self.trainer.step(ref_inputs, ref_labels, tar_inputs, tar_labels)




    def test_step(self, ref_inputs, ref_labels, tar_inputs, tar_labels):
        if self.ready_for_train:
            return self.validator.step(ref_inputs, ref_labels, tar_inputs, tar_labels)
