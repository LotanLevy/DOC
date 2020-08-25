
from abc import ABC,abstractmethod
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.applications import vgg16
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils


class TrainObject(ABC):

    def __init__(self, name, losses, metrics):
        self.name = name
        self.losses = dict()
        self.metrics = dict()
        self.trackers = dict()

        self.add_func_dict(losses, self.losses)
        self.add_func_dict(metrics, self.metrics, add_traker=False)

    def get_name(self, func_name):
        return self.name + "_" + func_name

    def add_func_dict(self, orig_dict, new_dict, add_traker=True):
        for func_name in orig_dict:
            new_dict[func_name] = orig_dict[func_name]
            if add_traker:
                self.trackers[self.get_name(func_name)] = tf.keras.metrics.Mean(name=self.get_name(func_name))
            else:
                self.trackers[self.get_name(func_name)] = new_dict[func_name]

    def update_state(self, func_name, value):
        self.trackers[self.get_name(func_name)].update_state(value)

    def get_state(self):
        result = dict()
        for tracker in self.trackers:
            result[tracker] = self.trackers[tracker].result().numpy()
        return result

    @abstractmethod
    def step(self, ref_inputs, ref_labels, tar_inputs, tar_labels):
        pass

    def reset(self):
        for name in self.trackers.keys():
            self.trackers[name].reset_states()




class Trainer(TrainObject):
    def __init__(self, name, losses, metrics, ref_model, tar_model, lambd, optimizer):
        super(Trainer, self).__init__(name, losses, metrics)
        self.ref_model = ref_model
        self.tar_model = tar_model
        self.lambd = lambd
        self.optimizer = optimizer

    def step(self, ref_inputs, ref_labels, tar_inputs, tar_labels):
        with tf.GradientTape(persistent=True) as tape:
            # Descriptiveness loss
            # ref_inputs = vgg16.preprocess_input(ref_inputs)

            ref_inputs = imagenet_utils.preprocess_input(ref_inputs)


            prediction = self.ref_model(ref_inputs, training=True)
            d_loss = self.losses["d_loss"](ref_labels, prediction)


            self.update_state("d_loss", d_loss)
            self.metrics["accuracy"].update_state(ref_labels, prediction)

            tar_inputs = imagenet_utils.preprocess_input(tar_inputs)

            #
            # # Compactness loss
            # prediction = self.tar_model(tar_inputs, training=True)
            # c_loss = self.losses["c_loss"](tar_labels, prediction)
            # self.update_state("c_loss", c_loss)

        d_gradients = tape.gradient(d_loss, self.ref_model.trainable_variables)


        # with tf.GradientTape() as tape:
        #
        #     # Compactness loss
        #     prediction = self.tar_model(tar_inputs, training=True)
        #     c_loss = self.losses["c_loss"](tar_labels, prediction)
        #     self.update_state("c_loss", c_loss)

        # c_gradients = tape.gradient(c_loss, self.tar_model.trainable_variables)

        # self.metrics["total"].update_state(d_loss * (1 - self.lambd) + c_loss * self.lambd)
        #
        # total_gradient = []
        # assert (len(d_gradients) == len(c_gradients))
        # for i in range(len(d_gradients)):
        #     total_gradient.append(d_gradients[i] * (1 - self.lambd) + c_gradients[i] * self.lambd)
        #
        # self.optimizer.apply_gradients(zip(total_gradient, self.ref_model.trainable_variables))

        self.optimizer.apply_gradients(zip(d_gradients, self.ref_model.trainable_variables))

        return self.get_state()




class Validator(TrainObject):
    def __init__(self, name, losses, metrics, ref_model, tar_model, lambd):
        super(Validator, self).__init__(name, losses, metrics)
        self.ref_model = ref_model
        self.tar_model = tar_model
        self.lambd = lambd


    def step(self, ref_inputs, ref_labels, tar_inputs, tar_labels):
        with tf.GradientTape(persistent=True) as tape:
            # Descriptiveness loss

            ref_inputs = vgg16.preprocess_input(ref_inputs)
            tar_inputs = vgg16.preprocess_input(tar_inputs)


            prediction = self.ref_model(ref_inputs, training=False)
            d_loss = self.losses["d_loss"](ref_labels, prediction)
            self.update_state("d_loss", d_loss)


            self.metrics["accuracy"].update_state(ref_labels, prediction)

            # # Compactness loss
            # prediction = self.tar_model(tar_inputs, training=False)
            # c_loss = self.losses["c_loss"](tar_labels, prediction)
            # self.update_state("c_loss", c_loss)



        # with tf.GradientTape() as tape:
        #     # Compactness loss
        #     prediction = self.tar_model(tar_inputs, training=False)
        #     c_loss = self.losses["c_loss"](tar_labels, prediction)
        #     self.update_state("c_loss", c_loss)

        # self.update_state("total", d_loss * (1 - self.lambd) + c_loss * self.lambd)


        return self.get_state()




