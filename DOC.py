import argparse
import tensorflow as tf
import numpy as np
import random
import nn_builder
import os
import datetime
import numpy as np
from train_test import AOC_helper
from tensorflow.python.keras.applications import vgg16
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils



from losses_and_metrices import  compactnessLoss, FeaturesLoss, CompactnesLoss2
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
    parser.add_argument("--alien_path", default=None,
                        help="The directory of the reference dataset")

    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--output_path", type=str, default=os.getcwd())

    parser.add_argument("--ref_aug", action='store_true')
    parser.add_argument("--tar_aug", action='store_true')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')


    parser.add_argument('--lambd', type=float, default=0.1,
                        help='lambda constant, the impact of the compactness loss')

    parser.add_argument('--test_size', type=int, default=100, help='number of alien and target examples for testing')



    parser.add_argument('--batch_size', '-b', type=int, default=2, help='number of batches')
    parser.add_argument('--print', type=int, default=50)
    parser.add_argument('--val_size', type=int, default=25)
    parser.add_argument('--iter', type=int, default=1000)





    return parser.parse_args()


def get_dataset(datagen, size, batch_size):
    images = []
    labels = []
    for _ in range(int(size / batch_size)):
        image, label = datagen.next()
        images.append(image)
        labels.append(label)


    return images, labels




def train(model, train_gens, steps_per_epoch, validation_data, validation_steps,
                               epochs, print_freq=10, summary_writer=None, aoc_helper=None, output_path=os.getcwd(), save_freq=100, batch_size=2):



    ref_train_gen, tar_train_gen = train_gens

    ref_val_gen, tar_val_gen = validation_data

    val_ref_inputs, val_ref_labels = get_dataset(ref_val_gen, 100, batch_size)

    val_tar_inputs, val_tar_labels = get_dataset(tar_train_gen, 100, batch_size)


    for epoch in range(epochs):

        count = 0



        for _ in range(steps_per_epoch):
            count += 1
            # print(count)

            # output = model.train_step(ref_inputs, ref_labels, tar_inputs, tar_labels)

            ref_inputs, ref_labels = ref_train_gen.next()
            tar_inputs, tar_labels = tar_train_gen.next()



            output = model.train_step(ref_inputs, ref_labels, tar_inputs, tar_labels)

            if count % print_freq == 0:



                for i in range(len(ref_inputs)):
                    val_output = model.test_step(val_ref_inputs[i], val_ref_labels[i], val_tar_inputs[i], val_tar_labels[i])


                output.update(val_output)

                if aoc_helper is not None:
                    roc_output = aoc_helper.get_roc_aoc(model)
                    output["aoc"] = roc_output[1]
                    output["target distance"] = roc_output[2]
                    output["alien distance"] = roc_output[3]



                print("iter: {}, {}".format(count, output))
                if summary_writer is not None:
                    with summary_writer.as_default():
                        for key in output:
                            tf.summary.scalar(key, output[key], step=count)

            if count % save_freq == 0:

                ckpt_path = os.path.join(output_path, "ckpts")
                checkpoint_path = "weights_after_{}_iterations".format(count)
                model.save_weights(os.path.join(ckpt_path, checkpoint_path))



def save_paths(file_name, dest, datagen):
    paths = datagen.filepaths
    with open(os.path.join(dest, file_name +".txt"), 'w') as f:
        for item in paths:
            f.write("%s\n" % item)








def main():
    tf.keras.backend.set_floatx('float32')
    # random.seed(1234)
    # np.random.seed(1234)
    # tf.random.set_seed(1234)

    args = get_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # data loaders #

    ref_gen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
    ref_classes = [str(i) for i in range(1000)]
    ref_train_datagen = ref_gen.flow_from_directory(args.ref_path, subset="training",
                                                    seed=123,
                                                    class_mode="categorical",
                                                    target_size=args.input_size,
                                                    batch_size=args.batch_size, classes=ref_classes)
    ref_val_datagen = ref_gen.flow_from_directory(args.ref_path, subset="validation",
                                                  seed=123,
                                                  class_mode="categorical",
                                                  target_size=args.input_size,
                                                  batch_size=args.batch_size, classes=ref_classes)

    tar_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2)

    tar_train_datagen = tar_gen.flow_from_directory(args.tar_path, subset="training",
                                                      seed=123,
                                                      class_mode="categorical",
                                                      target_size=args.input_size,
                                                      batch_size=args.batch_size)
    tar_val_datagen = tar_gen.flow_from_directory(args.tar_path, subset="validation",
                                                      seed=123,
                                                      class_mode="categorical",
                                                      target_size=args.input_size,
                                                      batch_size=args.batch_size)

    # ref_train_datagen, ref_val_datagen, tar_train_datagen, tar_val_datagen = create_generators(
    #                                                         args.ref_path, args.tar_path,
    #                                                         args.ref_aug, args.tar_aug,
    #                                                         args.input_size, args.batch_size)


    save_paths("ref_val", args.output_path, ref_val_datagen)
    save_paths("tar_val", args.output_path, tar_val_datagen)


    model = nn_builder.get_network(args.nntype, args.cls_num, args.input_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    templates_images = np.concatenate([tar_train_datagen.next()[0] for i in range(20)])

    features_loss = FeaturesLoss(templates_images, model)

    losses = {"d_loss": tf.keras.losses.CategoricalCrossentropy(),
              "c_loss": compactnessLoss(),
              "features_loss": features_loss}
    metrics = {"accuracy": tf.keras.metrics.CategoricalAccuracy, "total": tf.keras.metrics.Mean, "pred_val": tf.keras.metrics.Mean}

    aoc_helper = None

    if args.alien_path is not None:
        target_data = np.concatenate([tar_val_datagen.next()[0] for _ in range(int(args.test_size/args.batch_size))])

        alien_gen = tf.keras.preprocessing.image.ImageDataGenerator()

        alien_datagen = alien_gen.flow_from_directory(args.alien_path, subset="training",
                                                        seed=123,
                                                        class_mode="categorical",
                                                        target_size=args.input_size,
                                                    batch_size=target_data.shape[0])
        alien_data, _ = alien_datagen.next()
        aoc_helper = AOC_helper(templates_images, target_data, alien_data)



    log_dir = os.path.join(
        os.path.join(args.output_path, "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    summary_writer = tf.summary.create_file_writer(logdir=log_dir)

    model.set_ready_for_train(optimizer, args.lambd, losses=losses, metrics=metrics)

    train(model, (ref_train_datagen, tar_train_datagen), args.iter, (ref_val_datagen, tar_val_datagen), args.val_size,
          1, print_freq=args.print, summary_writer=summary_writer, aoc_helper=aoc_helper, output_path=args.output_path)







if __name__ == "__main__":
    main()

