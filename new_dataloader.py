
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

class DataIter:
    def __init__(self, paths, labels, batch_size, input_size, classes_num, shuffle=False):
        assert(len(paths) == len(labels))
        self.paths = paths
        self.classes_num = classes_num
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.cur_idx = 0
        self.indices = np.arange(len(self.paths)).astype(np.int)
        if shuffle:
            np.random.shuffle(self.indices)
        self.input_size = input_size
        self.cls2label = None

    def load_img(self, image_path):
        image = Image.open(image_path, 'r')
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(self.input_size, Image.NEAREST)
        image = np.array(image).astype(np.float32)
        return np.expand_dims(image, axis=0)

    def next(self):
        relevant_indices = self.indices[self.cur_idx: self.cur_idx + self.batch_size]
        self.cur_idx += self.batch_size
        images = np.concatenate([self.load_img(self.paths[i]) for i in relevant_indices])
        labels = self.labels[relevant_indices]
        return images, tf.keras.utils.to_categorical(labels, num_classes=self.classes_num)
        # return images, labels

    def set_cls2label_map(self, map):
        self.cls2label = map



def get_iterator_by_file_path(file_path, batch_size, input_size, classes_num):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        paths, labels = [], []
        for line in lines:
            path, label = line.split('$')
            paths.append(path)
            labels.append(int(label))
        return DataIter(paths, labels, batch_size, input_size, classes_num, shuffle=True)


def get_iterators_by_root_dir(root_dir, batch_size, input_size, split_val, classes_num, shuffle=True):
    dirs = os.listdir(root_dir)
    length = len(max(dirs, key=len))

    for dir in dirs:
        zeros = "0" * (length - len(dir))
        new_name = zeros + dir

        os.rename(os.path.join(root_dir, dir), os.path.join(root_dir, new_name))
        print("old {}, new {}".format(dir, new_name))

    paths = []
    labels = []
    cls2label = dict()
    label_idx = 0
    print(len(os.listdir(root_dir)))
    for sub_dir in os.listdir(root_dir):

        full_path = os.path.join(root_dir, sub_dir)
        if not os.path.isdir(full_path):
            continue
        cls2label[sub_dir] = label_idx
        for file in os.listdir(full_path):
            paths.append(os.path.join(full_path, file))
            labels.append(label_idx)
        label_idx += 1

    print(cls2label)


    assert len(paths) == len(labels)
    if len(cls2label) != classes_num:
        print("classes in directory doesn't match classes_num")

    X_train, X_test, y_train, y_test = train_test_split(paths, labels, test_size=split_val, shuffle=shuffle)

    print(list(zip(X_train, y_train)))

    train_iter = DataIter(X_train, y_train, batch_size, input_size, classes_num, shuffle=True)
    val_iter = DataIter(X_test, y_test, batch_size, input_size, classes_num, shuffle=True)

    train_iter.set_cls2label_map(cls2label)
    val_iter.set_cls2label_map(cls2label)
    return train_iter, val_iter