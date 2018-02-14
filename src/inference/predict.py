from src.data_preparation import dataset
from src.utils import helper
from nets import nets_factory
from sklearn.metrics import precision_recall_fscore_support as score

import sys
sys.path.append("/data/slim/models/research/slim/")
from datasets import flowers

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import csv


def predict_set(config):
    model_path = os.path.join('/data/checkpoints/flowers/', config.MODEL_NAME, str(config.TRAIN_LEARNING_RATE))
    if not os.path.exists(model_path):
        print("Model not exist: {}".format(model_path))
        exit()

    output_path = os.path.join(config.EVAL_OUTPUT, config.MODEL_NAME, "eval_results.csv")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # First create the dataset and load one batch
        valid_dataset = flowers.get_split('validation', config.TRAIN_TF_RECORDS)
        images, labels = dataset.load_batch(valid_dataset,
                                            batch_size=config.TRAIN_BATCH_SIZE,
                                            height=config.INPUT_HEIGHT,
                                            width=config.INPUT_WIDTH,
                                            epochs=1)
        # Create the model inference
        net_fn = nets_factory.get_network_fn(
            config.PRETAIN_MODEL,
            dataset.num_classes,
            weight_decay=config.L2_WEIGHT_DECAY,
            is_training=False)

        logits, end_points = net_fn(images)
        predictions = tf.argmax(end_points['Predictions'], 1)

        # Define the scopes that you want to exclude for restoration
        variables_to_restore = slim.get_variables_to_restore()

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess, open(output_path, 'w') as f:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            print("Restore model from: {}".format(model_path))

            writer = csv.writer(f)
            writer.writerow(["y_true", "preds"])
            while True:
                y_true, preds = sess.run([labels, predictions])
                results = zip(y_true, preds)
                writer.writerows("{}, {}".format(row[0], row[1]) for row in results)
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
