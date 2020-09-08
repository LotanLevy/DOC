
import tensorflow as tf
from tensorflow.python.keras.applications import vgg16
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.python.keras.models import Model
from PIL import Image


np.random.seed(123)




ref_path = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\datasets\\imagenet_val_splitted"
input_size = (224,224)
batch_size = 16
steps_per_epoch = 1000
validation_steps = 10
print_freq = 10
initial_epochs = 10
classes_num = 1000



def read_image(image_path, input_size):
    image = load_img(image_path, target_size=input_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

class DataIter:
    def __init__(self, paths, labels, batch_size, input_size, shuffle=False):
        assert(len(paths) == len(labels))
        self.paths = paths
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.cur_idx = 0
        self.indices = np.arange(len(self.paths)).astype(np.int)
        if shuffle:
            np.random.shuffle(self.indices)
        self.input_size = input_size

    def load_img(self, image_path):
        image = Image.open(image_path, 'r')
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(input_size, Image.NEAREST)
        image = np.array(image).astype(np.float32)
        return np.expand_dims(image, axis=0)

    def next(self):
        relevant_indices = self.indices[self.cur_idx: self.cur_idx + self.batch_size]
        self.cur_idx += self.batch_size
        images = np.concatenate([self.load_img(self.paths[i]) for i in relevant_indices])
        labels = self.labels[relevant_indices]
        return images, tf.keras.utils.to_categorical(labels, num_classes=classes_num)
        # return images, labels




def get_iterator(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        paths, labels = [], []
        for line in lines:
            path, label = line.split('$')
            paths.append(path)
            labels.append(int(label))
        return DataIter(paths, labels, batch_size, input_size, shuffle=True)


# ref_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#             validation_split=0.2)
# ref_classes = [str(i) for i in range(1000)]
# ref_train_datagen = ref_gen.flow_from_directory(ref_path, subset="training",
#                                                   seed=123,
#                                                   class_mode="categorical",
#                                                   target_size=input_size,
#                                                   batch_size=batch_size, classes=ref_classes)
# ref_val_datagen = ref_gen.flow_from_directory(ref_path, subset="validation",
#                                                   seed=123,
#                                                   class_mode="categorical",
#                                                   target_size=input_size,
#                                                   batch_size=batch_size, classes=ref_classes)


ref_train_datagen = get_iterator("C:/Users/lotan/Documents/studies/Affordances/datasets/imagenet_files/train.txt")
ref_val_datagen = get_iterator("C:/Users/lotan/Documents/studies/Affordances/datasets/imagenet_files/val.txt")


model = vgg16.VGG16(weights="imagenet")
loss_fn = tf.keras.losses.CategoricalCrossentropy()
accuracy_func_train = tf.keras.metrics.CategoricalAccuracy()
loss_tracker_train = tf.keras.metrics.Mean(name="loss")

accuracy_func_val = tf.keras.metrics.CategoricalAccuracy()
loss_tracker_val = tf.keras.metrics.Mean(name="loss")

for layer in model.layers[:19]:
    layer.trainable = False
    print(layer.name)

model.summary()


base_learning_rate = 0.0001
optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate)


for step in range(steps_per_epoch):

    # Iterate over the batches of the dataset.

        (x_batch_train, y_batch_train) = ref_train_datagen.next()


        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            assert x_batch_train.shape[0] == batch_size

            import matplotlib.pyplot as plt


            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            proc = imagenet_utils.preprocess_input(np.copy(x_batch_train))
            preds = model(proc, training=True)

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, preds)
            accuracy_func_train.update_state(y_batch_train, preds)
            loss_tracker_train.update_state(loss_value)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % print_freq == 0:

            # plt.figure()
            # plt.imshow(x_batch_train[0].astype(np.int))
            # plt.title(np.argmax(y_batch_train[0]))
            # plt.show()


            for i in range(validation_steps):
                (x_batch_val, y_batch_val) = ref_train_datagen.next()
                with tf.GradientTape() as tape:
                    assert x_batch_val.shape[0] == batch_size
                    proc = imagenet_utils.preprocess_input(np.copy(x_batch_val))

                    preds = model(proc, training=False)

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(y_batch_val, preds)

                    accuracy_func_val.update_state(y_batch_val, preds)
                    loss_tracker_val.update_state(loss_value)


            result = dict()
            result["step"] = step
            result["train loss"] = loss_tracker_train.result().numpy()
            result["train accuracy"] = accuracy_func_train.result().numpy()
            result["val loss"] = loss_tracker_val.result().numpy()
            result["val accuracy"] = accuracy_func_val.result().numpy()
            print(result)

