
import tensorflow as tf
from Networks.NNInterface import NNInterface
from tensorflow.python.keras.applications import vgg16
import numpy as np



class Simple(NNInterface):
    def __init__(self):
        super().__init__()
        self.model = vgg16.VGG16(weights='imagenet')

        self.is_compiled = False
        self.loss = None
        self.optimizer = None
        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="train_loss")

        self.train_accuracy_tracker =tf.keras.metrics.CategoricalAccuracy()
        self.val_accuracy_tracker =tf.keras.metrics.CategoricalAccuracy()

    def freeze_conv_layers(self):
        for layer in self.model.layers[:20]:
            layer.trainable = False
        msg = "trainable statuse:\n"
        for layer in self.model.layers:
            msg += "layer {} is trainable {}\n".format(layer.name, layer.trainable)
        print(msg)



    def compile_model(self, optimizer, loss):
        self.loss = loss
        self.optimizer = optimizer
        self.is_compiled = True


    def call(self, x, training=True):
        self.model(x)

    def train_step(self, inputs, labels):
        if not self.is_compiled:
            print("The model is not compiled")
            return

        with tf.GradientTape() as tape:
            # Descriptiveness loss
            # proc_inputs = vgg16.preprocess_input(np.copy(inputs))

            prediction = self.model(inputs, training=True)
            d_loss = self.loss(labels, prediction)

        d_gradients = tape.gradient(d_loss, self.model.trainable_variables)

        self.train_loss_tracker.update_state(d_loss)
        self.train_accuracy_tracker.update_state(labels, prediction)
        self.optimizer.apply_gradients(zip(d_gradients, self.model.trainable_variables))
        return {"train_loss": self.train_loss_tracker.result().numpy(),
                "train_accuracy": self.train_accuracy_tracker.result().numpy()}










    def test_step(self, inputs, labels):
        if not self.is_compiled:
            print("The model is not compiled")
            return

        with tf.GradientTape() as tape:
            # Descriptiveness loss
            # proc_inputs = vgg16.preprocess_input(np.copy(inputs))

            prediction = self.model(np.copy(inputs), training=False)
            d_loss = self.loss(labels, prediction)

        self.val_loss_tracker.update_state(d_loss)
        self.val_accuracy_tracker.update_state(labels, prediction)
        return {"val_loss": self.train_loss_tracker.result().numpy(),
                "val_accuracy": self.train_accuracy_tracker.result().numpy()}
