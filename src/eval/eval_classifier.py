import math
import datetime
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from src.data_preparation import dataset
from src.utils import helper
from nets import nets_factory
from sklearn.metrics import precision_recall_fscore_support as score

sys.path.append("/data/slim/models/research/slim/")
from datasets import flowers

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == '__main__':
    slim = tf.contrib.slim

    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c',
                        dest="config_filename", type=str, required=True,
                        help='the config file name must be provide')
    args = parser.parse_args()

    config = helper.parse_config_file(args.config_filename)
    eval_summary_dir = os.path.join("/data/summary/flowers/", config.MODEL_NAME, str(config.TRAIN_LEARNING_RATE), "eval")
    checkpoint_dir = os.path.join('/data/checkpoints/flowers/', config.MODEL_NAME, str(config.TRAIN_LEARNING_RATE))

    # This might take a few minutes.
    with tf.Graph().as_default():
        summary_ops = []

        # Load the data
        train_dataset = flowers.get_split('validation', config.TRAIN_TF_RECORDS)
        images, labels = dataset.load_batch(train_dataset, config.EVAL_BATCH_SIZE,
                                            height=config.INPUT_HEIGHT, width=config.INPUT_WIDTH, is_training=False)

        summary_ops.append(tf.summary.image('images/val', images))

        # Create the model inference
        net_fn = nets_factory.get_network_fn(config.PRETAIN_MODEL, dataset.num_classes, is_training=False)
        logits, end_points = net_fn(images)
        predictions = tf.argmax(logits, 1)

        precision, recall, f1, _ = score(labels, predictions)
        summary_ops.append(tf.summary.scalar('precision', np.mean(precision)))
        summary_ops.append(tf.summary.scalar('Recall', np.mean(recall)))

        # Specify the loss function:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        total_loss = tf.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        summary_ops.append(tf.summary.scalar('losses/total_loss', total_loss))

        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        summary_ops.append(tf.summary.scalar('accuracy', accuracy))

        # Choose the metrics to compute:
        # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        #     "accuracy": tf.metrics.accuracy(labels, predictions),
        #     # 'precision': slim.metrics.streaming_precision(predictions, labels),
        #     # 'Recall@1': slim.metrics.streaming_recall_at_k(logits, labels, 1)
        # })

        # Create the summary ops such that they also print out to std output:
        # for metric_name, metric_value in names_to_values.iteritems():
        #     print(metric_name)
        #     op = tf.summary.scalar(metric_name, metric_value)
        #     op = tf.Print(op, [metric_value], metric_name)
        #     summary_ops.append(op)

        num_examples = 200
        num_batches = math.ceil(num_examples / config.EVAL_BATCH_SIZE)

        # Setup the global step.
        slim.get_or_create_global_step()

        # How often to run the evaluation.
        eval_interval_secs = 10
        slim.evaluation.evaluation_loop(
            '',
            checkpoint_dir,
            eval_summary_dir,
            num_evals=num_batches,
            #eval_op=names_to_updates.values(),
            summary_op=tf.summary.merge(summary_ops),
            eval_interval_secs=eval_interval_secs)
