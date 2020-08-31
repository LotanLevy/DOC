
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
            # new_dict[func_name] = orig_dict[func_name]()
            if add_traker:
                new_dict[func_name] = orig_dict[func_name]

                self.trackers[self.get_name(func_name)] = tf.keras.metrics.Mean(name=self.get_name(func_name))
            else:
                new_dict[func_name] = orig_dict[func_name]()

                self.trackers[self.get_name(func_name)] = new_dict[func_name]

    def update_metric_state(self, func_name, *args):
        self.metrics[func_name].update_state(*args)

    def update_loss_state(self, func_name, *args):
        loss = self.losses[func_name](*args)
        self.trackers[self.get_name(func_name)].update_state(loss)
        return loss

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
            ref_proc_inputs = vgg16.preprocess_input(np.copy(ref_inputs))
            tar_proc_inputs = vgg16.preprocess_input(np.copy(tar_inputs))

            prediction = self.ref_model(ref_proc_inputs, training=True)
            d_loss = self.update_loss_state("d_loss", ref_labels, prediction)
            self.update_metric_state("accuracy", ref_labels, prediction)

            # d_loss = self.losses["d_loss"](ref_labels, prediction)
            #
            # self.update_state("d_loss", d_loss)
            # self.metrics["accuracy"].update_state(ref_labels, prediction)


            print(np.argmax(ref_labels, axis=1))
            print(np.argmax(prediction, axis=1), np.max(prediction, axis=1))


        #
        # with tf.GradientTape() as tape:

            # fig, axs = plt.subplots(2)
            # axs[0].imshow(tar_inputs[0].astype(np.int))
            # axs[1].imshow(tar_inputs[1].astype(np.int))
            # plt.show()

            # Compactness loss
            prediction = self.tar_model(tar_proc_inputs, training=True)
            c_loss = self.update_loss_state("c_loss", tar_labels, prediction)
            self.update_loss_state("features_loss", tar_labels, prediction)
            # c_loss = self.losses["c_loss"](tar_labels, prediction)
            # self.update_state("c_loss", c_loss)

            # features_loss = self.losses["features_loss"](tar_labels, prediction)
            # self.update_state("features_loss", features_loss)

        d_gradients = tape.gradient(d_loss, self.ref_model.trainable_variables)
        c_gradients = tape.gradient(c_loss, self.tar_model.trainable_variables)

        self.update_metric_state("total", d_loss * (1 - self.lambd) + c_loss * self.lambd)

        # self.metrics["total"].update_state(d_loss * (1 - self.lambd) + c_loss * self.lambd)
        #
        total_gradient = []
        assert (len(d_gradients) == len(c_gradients))
        for i in range(len(d_gradients)):
            total_gradient.append(d_gradients[i] * (1 - self.lambd) + c_gradients[i] * self.lambd)

        assert self.ref_model.trainable_variables == self.tar_model.trainable_variables

        self.optimizer.apply_gradients(zip(total_gradient, self.ref_model.trainable_variables))


        # self.optimizer.apply_gradients(zip(d_gradients, self.ref_model.trainable_variables))

        # self.optimizer.apply_gradients(zip(d_gradients, self.ref_model.trainable_variables))

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
            #
            ref_proc_inputs = vgg16.preprocess_input(np.copy(ref_inputs))
            tar_proc_inputs = vgg16.preprocess_input(np.copy(tar_inputs))


            prediction = self.ref_model(ref_proc_inputs, training=False)

            d_loss = self.update_loss_state("d_loss", ref_labels, prediction)
            self.update_metric_state("accuracy", ref_labels, prediction)
            # d_loss = self.losses["d_loss"](ref_labels, prediction)
            # self.update_state("d_loss", d_loss)

            # print(np.argmax(ref_labels, axis=1))
            # print(np.argmax(prediction, axis=1), np.max(prediction, axis=1))



            #
            # self.metrics["accuracy"].update_state(ref_labels, prediction)



        # with tf.GradientTape() as tape:

            # Compactness loss
            prediction = self.tar_model(tar_proc_inputs, training=False)
            c_loss = self.update_loss_state("c_loss", tar_labels, prediction)
            self.update_loss_state("features_loss", tar_labels, prediction)
            # c_loss = self.losses["c_loss"](tar_labels, prediction)
            # self.update_state("c_loss", c_loss)
            # features_loss = self.losses["features_loss"](tar_labels, prediction)
            # self.update_state("features_loss", features_loss)

            # fig, axs = plt.subplots(2)
            # axs[0].imshow(tar_inputs[0].astype(np.int))
            # axs[1].imshow(tar_inputs[1].astype(np.int))
            # plt.show()



        # self.metrics["total"].update_state(d_loss * (1 - self.lambd) + c_loss * self.lambd)

        self.update_metric_state("total", d_loss * (1 - self.lambd) + c_loss * self.lambd)





        # self.update_state("total", d_loss * (1 - self.lambd) + c_loss * self.lambd)


        return self.get_state()




