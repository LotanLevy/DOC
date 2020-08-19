# from tensorflow.keras import Model
import tensorflow as tf

class NNInterface(tf.keras.Model):
    """
    Interface of a network.
    Only classes that inherit from it can be created by the program.
    """
    def __init__(self):
        super(NNInterface, self).__init__()

