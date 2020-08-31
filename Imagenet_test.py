import argparse
import tensorflow as tf
import numpy as np
import random
import nn_builder
import os
import datetime
from tensorflow.python.keras.applications import vgg16
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils
from dataloader_2 import DataLoader



from losses_and_metrices import  compactnessLoss
from dataloader import create_generators
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

def get_args():
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nntype', default="Simple", choices=["Simple"])

    parser.add_argument('--cls_num', type=int, default=1000)
    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))

    parser.add_argument("--path", required=True,
                        help="The directory of the reference dataset")


    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--output_path", type=str, default=os.getcwd())

    parser.add_argument("--ref_aug", action='store_true')
    parser.add_argument("--tar_aug", action='store_true')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')


    parser.add_argument('--lambd', type=float, default=0.1,
                        help='lambda constant, the impact of the compactness loss')


    parser.add_argument('--batch_size', '-b', type=int, default=32, help='number of batches')
    parser.add_argument('--print', type=int, default=10)
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--iter', type=int, default=800)

    return parser.parse_args()


def train(model, dataloader, steps_per_epoch, validation_steps,
                               epochs, print_freq=10):

    for epoch in range(epochs):
        count = 0
        for _ in range(steps_per_epoch):
            count += 1
            inputs, labels = dataloader.read_batch(2, "train")

            output = model.train_step(inputs, labels)
            print(output)

            if count % print_freq == 0:
                inputs, labels = dataloader.read_batch(2, "val")

                output = model.test_step(inputs, labels)
                print(output)



                print("iter: {}, {}".format(count, output))




def main():
    tf.keras.backend.set_floatx('float32')
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    args = get_args()

    dataloader = DataLoader(os.path.join(args.path, "train.txt"),
                            os.path.join(args.path, "val.txt"),
                            os.path.join(args.path, "test.txt"), args.cls_num, args.input_size,
                            name="dataloader", output_path=args.output_path)

    # data loaders #
    # ref_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     preprocessing_function=vgg16.preprocess_input,
    #     validation_split=0.2)
    # classes = [str(i) for i in range(1000)]
    # train_datagen = ref_gen.flow_from_directory(args.path, subset="training",
    #                                                 seed=123,
    #                                                 shuffle=True,
    #                                                 class_mode="categorical",
    #                                                 target_size=args.input_size,
    #                                                 batch_size=args.batch_size, classes=classes)
    # val_datagen = ref_gen.flow_from_directory(args.path, subset="validation",
    #                                               seed=123,
    #                                               shuffle=True,
    #                                               class_mode="categorical",
    #                                               target_size=args.input_size,
    #                                               batch_size=args.batch_size, classes=classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model = nn_builder.get_network(args.nntype)
    model.freeze_conv_layers()
    model.compile_model(optimizer, loss)


    # model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    #
    # model.fit_generator(train_datagen, validation_data=val_datagen, steps_per_epoch=50,
	# epochs=1)



    train(model, dataloader, 20, 20,
          1, print_freq=10)

if __name__ == "__main__":
    main()

