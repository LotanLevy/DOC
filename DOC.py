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



from losses_and_metrices import  compactnessLoss
from dataloader import create_generators
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

def get_args():
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nntype', default="DOCModel", choices=["DOCModel"])

    parser.add_argument('--cls_num', type=int, default=1000)
    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))

    parser.add_argument("--ref_path", required=True,
                        help="The directory of the reference dataset")
    parser.add_argument("--tar_path", required=True,
                        help="The directory of the target dataset")

    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--output_path", type=str, default=os.getcwd())

    parser.add_argument("--ref_aug", action='store_true')
    parser.add_argument("--tar_aug", action='store_true')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')


    parser.add_argument('--lambd', type=float, default=0.1,
                        help='lambda constant, the impact of the compactness loss')


    parser.add_argument('--batch_size', '-b', type=int, default=2, help='number of batches')
    parser.add_argument('--print', type=int, default=50)
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--iter', type=int, default=800)





    return parser.parse_args()

def train(model, train_gens, steps_per_epoch, validation_data, validation_steps,
                               epochs, print_freq=10):
    ref_train_gen, tar_train_gen = train_gens

    ref_val_gen, tar_val_gen = validation_data

    for epoch in range(epochs):

        count = 0

        for _ in range(steps_per_epoch):
            count += 1
            ref_inputs, ref_labels = ref_train_gen.next()
            tar_inputs, tar_labels = tar_train_gen.next()
            output = model.train_step(ref_inputs, ref_labels, tar_inputs, tar_labels)

            if count % print_freq == 0:

                print("iter: {}, {}".format(count, output))

                if count % 2 * print_freq == 0:

                    for _ in range(validation_steps):
                        ref_inputs, ref_labels = ref_val_gen.next()
                        tar_inputs, tar_labels = tar_val_gen.next()
                        output = model.test_step(ref_inputs, ref_labels, tar_inputs, tar_labels)
                    print("iter: {}, {}".format(count, output))

                    model.on_validation_epoch_end()







def main():
    tf.keras.backend.set_floatx('float32')
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    args = get_args()

    # data loaders #

    ref_train_datagen, ref_val_datagen, tar_train_datagen, tar_val_datagen = create_generators(
                                                            args.ref_path, args.tar_path,
                                                            args.ref_aug, args.tar_aug,
                                                            args.input_size, args.batch_size)



    # build the network #
    model = nn_builder.get_network(args.nntype, args.cls_num, args.input_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    losses = {"d_loss":tf.keras.losses.CategoricalCrossentropy(),
              "c_loss": compactnessLoss()}


    metrics = {"accuracy": tf.keras.metrics.CategoricalAccuracy(), "total":tf.keras.metrics.Mean(name="total")}
    model.set_ready_for_train(optimizer, args.lambd, losses=losses, metrics=metrics)



    log_dir = os.path.join(
        os.path.join(args.output_path, "amazon_logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    chackpoint_path = os.path.join(os.path.join(args.output_path, "checkpoint"))

    checkpoint = ModelCheckpoint(chackpoint_path, monitor='val_accuracy', save_best_only=True,
                                 save_weights_only=False, mode='max', verbose=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

    csv_logger = CSVLogger(os.path.join(args.output_path, 'log.csv'), append=True, separator=';')

    train(model, (ref_train_datagen, tar_train_datagen), args.iter, (ref_val_datagen, tar_val_datagen), args.val_size,
          1, print_freq=args.print)






    # hist = model.fit_generator(generator=train_datagen, steps_per_epoch=50, validation_data=val_datagen, validation_steps=10,
    #                            epochs=10,
    #                            workers=2, use_multiprocessing=True)





if __name__ == "__main__":
    main()

