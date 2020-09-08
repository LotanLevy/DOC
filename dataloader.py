
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.keras.applications import vgg16

from tensorflow.keras.applications import imagenet_utils


def read_image(image_path, input_size):
    image = load_img(image_path, target_size=input_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image

#
# class DirectoryLoader:
#     def __init__(self, directory_path, input_size):
#         self.labels = []
#         self.images = []
#         self.input_shape = (input_size[0], input_size[1], 3)
#
#     def load(self, directory_path):
#         classes = os.listdir(directory_path)
#         labels = np.arange(len(classes))
#         self._cls_to_label = dict(zip(classes, labels))
#         self._label_to_cls = dict(zip(labels, classes))
#
#         for cls in classes:
#             cls_dir_path = os.path.join(directory_path, cls)
#             label = self._cls_to_label[cls]
#             images_names = os.listdir(cls_dir_path)
#             for name in images_names:
#                 image_path = os.path.join(cls_dir_path, name)



class DOCSequence(tf.keras.utils.Sequence):

    def __init__(self, ref_generator, tar_generator, batch_size, shuffle=True):
        self.ref_generator, self.tar_generator = ref_generator, tar_generator
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return max(len(self.ref_generator), len(self.tar_generator))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(max(len(self.ref_generator), len(self.tar_generator)))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        ref_data, ref_labels = self.ref_generator[idx % len(self.ref_generator)]
        tar_data, tar_labels = self.tar_generator[idx % len(self.tar_generator)]
        return ref_data, ref_labels, tar_data, tar_labels


def get_gen(to_aug):
    if to_aug:
        return tf.keras.preprocessing.image.ImageDataGenerator(
                                    rotation_range=20,
                                    zoom_range=0.15,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.15,
                                    horizontal_flip=True,
                                    fill_mode="nearest",
                                    validation_split=0.2)
    else:
        return tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=vgg16.preprocess_input,
            validation_split=0.2)






def get_directory_iterator(gen, name, input_size, batch_size, dir_path):
    return gen.flow_from_directory(dir_path, subset=name,
                                                      seed=123,
                                                      class_mode="categorical",
                                                      target_size=input_size,
                                                      batch_size=batch_size)

def create_generators(ref_path, tar_path, ref_aug, tar_aug, input_size, batch_size):
    ref_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            validation_split=0.2)
    ref_classes = [str(i) for i in range(1000)]
    ref_train_datagen = ref_gen.flow_from_directory(ref_path, subset="training",
                                                      seed=123,
                                                      class_mode="categorical",
                                                      target_size=input_size,
                                                      batch_size=batch_size, classes=ref_classes)
    ref_val_datagen = ref_gen.flow_from_directory(ref_path, subset="validation",
                                                      seed=123,
                                                      class_mode="categorical",
                                                      target_size=input_size,
                                                      batch_size=batch_size, classes=ref_classes)
    # ref_train_datagen = get_directory_iterator(ref_gen, "training", input_size, batch_size, ref_path)
    # ref_val_datagen = get_directory_iterator(ref_gen, "validation", input_size, batch_size, ref_path)

    tar_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2)
    tar_train_datagen = get_directory_iterator(tar_gen, "training", input_size, batch_size, tar_path)
    tar_val_datagen = get_directory_iterator(tar_gen, "validation", input_size, batch_size, tar_path)

    return ref_train_datagen, ref_val_datagen, tar_train_datagen, tar_val_datagen

