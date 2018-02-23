from src.data_preparation import dataset
from src.utils import helper
from nets import nets_factory
from sklearn.metrics import precision_recall_fscore_support as score

import sys
sys.path.append("/data/slim/models/research/slim/")
from datasets import flowers
from itertools import izip

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import csv


def predict_set(config_):
    model_path = os.path.join('/data/checkpoints/flowers/', config_.MODEL_NAME, str(config_.TRAIN_LEARNING_RATE))
    if not os.path.exists(model_path):
        print("Model not exist: {}".format(model_path))
        exit()

    output_path = os.path.join(config_.EVAL_OUTPUT, config_.MODEL_NAME)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    eval_output = os.path.join(output_path, "eval_results.csv")

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # First create the dataset and load one batch
        valid_dataset = flowers.get_split('validation', config_.TRAIN_TF_RECORDS)
        images, labels = dataset.load_batch(valid_dataset,
                                            batch_size=config_.TRAIN_BATCH_SIZE,
                                            height=config_.INPUT_HEIGHT,
                                            width=config_.INPUT_WIDTH,
                                            epochs=1)
        # Create the model inference
        net_fn = nets_factory.get_network_fn(
            config_.PRETAIN_MODEL,
            dataset.num_classes,
            is_training=False)

        _, end_points = net_fn(images)
        predictions = tf.argmax(end_points['Predictions'], 1)

        # Define the scopes that you want to exclude for restoration
        variables_to_restore = slim.get_variables_to_restore()

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        checkpoint_path = tf.train.latest_checkpoint(model_path)
        init_fn = slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore)

        with tf.Session() as sess, open(eval_output, 'w') as f:
            print("writing results to {}".format(eval_output))
            init_fn(sess)
            print("Restore model from: {}".format(model_path))

            sess.run(tf.initialize_local_variables())

            writer = csv.writer(f)
            writer.writerow(["y_true", "preds"])

            try:
                with slim.queues.QueueRunners(sess):
                    while True:
                        y_true, preds = sess.run([labels, predictions])
                        writer.writerows(zip(y_true, preds))
                        print("precessing {} records".format(str(y_true.shape[0])))
            except tf.errors.OutOfRangeError:
                print('')
        print("Prediction finished.")


if __name__ == '__main__':
    slim = tf.contrib.slim

    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c',
                        dest="config_filename", type=str, required=True,
                        help='the config file name must be provide')
    args = parser.parse_args()

    config = helper.parse_config_file(args.config_filename)
    predict_set(config)

