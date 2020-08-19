
import argparse
import tensorflow as tf
import numpy as np
import random
import nn_builder
import os
import datetime

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

    parser.add_argument('--lambd', type=float, default=0.1,
                        help='lambda constant, the impact of the compactness loss')





    parser.add_argument('--batch_size', '-b', type=int, default=2, help='number of batches')



    return parser.parse_args()


def main():
    tf.keras.backend.set_floatx('float32')
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    args = get_args()

    # data loaders #

    train_datagen, val_datagen = create_generators(args.ref_path, args.tar_path,
                                                     args.ref_aug, args.tar_aug,
                                                   args.input_size, args.batch_size)

    # build the network #
    model = nn_builder.get_network(args.nntype, args.cls_num, args.input_size)

    optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)

    losses_and_metrics = {"d_loss":tf.keras.losses.SparseCategoricalCrossentropy(),
              "c_loss": compactnessLoss(),
              "accuracy": tf.keras.metrics.Accuracy()}
    model.set_losses_and_metrics(losses_and_metrics, args.lambd)

    model.compile(optimizer=optimizer)


    log_dir = os.path.join(
        os.path.join(args.output_path, "amazon_logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    chackpoint_path = os.path.join(os.path.join(args.output_path, "checkpoint"))

    checkpoint = ModelCheckpoint(chackpoint_path, monitor='val_accuracy', save_best_only=True,
                                 save_weights_only=False, mode='max', verbose=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

    csv_logger = CSVLogger(os.path.join(args.output_path, 'log.csv'), append=True, separator=';')




    hist = model.fit_generator(generator=train_datagen, steps_per_epoch=50, validation_data=val_datagen, validation_steps=10,
                               epochs=10,
                               workers=2, use_multiprocessing=True)





if __name__ == "__main__":
    main()

