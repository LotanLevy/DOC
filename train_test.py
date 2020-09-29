
from abc import ABC,abstractmethod
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.applications import vgg16
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import roc_curve, auc
from losses_and_metrices import FeaturesLoss



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
    def __init__(self, name, losses, metrics, model, lambd, optimizer):
        super(Trainer, self).__init__(name, losses, metrics)
        self.model = model
        self.lambd = lambd
        self.optimizer = optimizer


    def step(self, ref_inputs, ref_labels, tar_inputs, tar_labels):

        #compactness step


        with tf.GradientTape(persistent=True) as tape:


            # Descriptiveness loss
            # ref_proc_inputs = vgg16.preprocess_input(np.copy(ref_inputs))
            # tar_proc_inputs = vgg16.preprocess_input(np.copy(tar_inputs))

            prediction = self.model(ref_inputs, training=True, ref_state=True)
            d_loss = self.update_loss_state("d_loss", ref_labels, prediction)
            self.update_metric_state("accuracy", ref_labels, prediction)

            self.update_metric_state("pred_val", np.mean(np.max(prediction, axis=1)))

            # Compactness loss
            prediction = self.model(tar_inputs, training=True, ref_state=False)
            c_loss = self.update_loss_state("c_loss", tar_labels, prediction)
            # self.update_loss_state("features_loss", tar_labels, prediction)

        d_gradients = tape.gradient(d_loss, self.model.trainable_variables)
        c_gradients = tape.gradient(c_loss, self.model.trainable_variables)

        self.update_metric_state("total", d_loss * (1 - self.lambd) + c_loss * self.lambd)

        total_gradient = []
        assert (len(d_gradients) == len(c_gradients))
        for i in range(len(d_gradients)):
            total_gradient.append(d_gradients[i] * (1 - self.lambd) + c_gradients[i] * self.lambd)


        self.optimizer.apply_gradients(zip(total_gradient, self.model.trainable_variables))


        # self.optimizer.apply_gradients(zip(d_gradients, self.model.trainable_variables))




        # prediction = self.model(ref_proc_inputs, training=False)






        return self.get_state()




class Validator(TrainObject):
    def __init__(self, name, losses, metrics, model, lambd):
        super(Validator, self).__init__(name, losses, metrics)
        self.model = model
        self.lambd = lambd


    def step(self, ref_inputs, ref_labels, tar_inputs, tar_labels):
        with tf.GradientTape(persistent=True) as tape:

            # ref_proc_inputs = vgg16.preprocess_input(np.copy(ref_inputs))
            # tar_proc_inputs = vgg16.preprocess_input(np.copy(tar_inputs))


            prediction = self.model(ref_inputs, training=False, ref_state=True)

            d_loss = self.update_loss_state("d_loss", ref_labels, prediction)
            self.update_metric_state("accuracy", ref_labels, prediction)

            self.update_metric_state("pred_val", np.mean(np.max(prediction, axis=1)))



            # print(np.argmax(ref_labels, axis=1))
            # print(np.argmax(prediction, axis=1), np.max(prediction, axis=1))

        # with tf.GradientTape() as tape:

            #Compactness loss
            prediction = self.model(tar_inputs, training=False, ref_state=False)
            c_loss = self.update_loss_state("c_loss", tar_labels, prediction)
            # self.update_loss_state("features_loss", tar_labels, prediction)

        self.update_metric_state("total", d_loss * (1 - self.lambd) + c_loss * self.lambd)


        return self.get_state()




class AOC_helper:

    def __init__(self, templates, targets, aliens, cls_num, batch_size):
        self.templates = templates
        self.targets = targets
        self.aliens = aliens
        self.cls_num = cls_num
        self.batch_size = batch_size


    def get_roc_aoc(self, model):
        optimal_cutoff, roc_auc, target_scores, alien_scores = self.get_roc_aoc_with_scores(model)

        return optimal_cutoff, roc_auc, np.mean(target_scores), np.mean(alien_scores)

    def get_roc_aoc_with_scores(self, model):
        feature_loss = FeaturesLoss(self.templates, model, self.cls_num, self.batch_size)

        target_num = len(self.targets)
        alien_num = len(self.aliens)

        scores = np.zeros(target_num + alien_num)
        labels = np.zeros(target_num + alien_num)

        preds = model.target_call(self.targets, training=False)
        scores[:target_num] = feature_loss(None, preds)
        labels[:target_num] = np.zeros(target_num)

        preds = model.target_call(self.aliens, training=False)
        scores[target_num:] = feature_loss(None, preds)
        labels[target_num:] = np.ones(alien_num)

        fpr, tpr, thresholds = roc_curve(labels, -scores, 0)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_cutoff = thresholds[optimal_idx]
        return optimal_cutoff, roc_auc, scores[:target_num], scores[target_num:]


