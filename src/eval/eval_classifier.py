import math
import datetime
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from src.data_preparation import dataset

import sys, os
sys.path.append("/data/slim/models/research/slim/")
from datasets import flowers
from preprocessing import inception_preprocessing


if __name__ == '__main__':
    slim = tf.contrib.slim

    model_name = "slim_inception_v1"
    l_rate = 0.001

    CHECKPOINT_DIR = '/data/checkpoints/flowers/'
    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name, str(l_rate))

    VAL_SUMMARY_DIR = "/data/summary/flowers/val"
    log_dir = os.path.join(VAL_SUMMARY_DIR, model_name, str(l_rate), datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    flowers_data_dir = "/data/flowers"
    batch_size = 10

    image_size = inception.inception_v1.default_image_size

    # This might take a few minutes.
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        summary_ops = []

        # Load the data
        train_dataset = flowers.get_split('validation', flowers_data_dir)
        images, labels = dataset.load_batch(train_dataset, batch_size,
                                            height=image_size, width=image_size, is_training=False)

        summary_ops.append(tf.summary.image('images/val', images))

        # Create the model:
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v1(images, num_classes=5, is_training=False)
            predictions = tf.argmax(logits, 1)

        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, 5)
        val_loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        summary_ops.append(tf.summary.scalar('losses/total_loss', total_loss))

        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            "eval/accuracy": slim.metrics.streaming_accuracy(predictions, labels),
            'eval/precision': slim.metrics.streaming_precision(predictions, labels),
            'eval/Recall@1': slim.metrics.streaming_recall_at_k(logits, labels, 1)
        })

        # Create the summary ops such that they also print out to std output:
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        num_examples = 10
        num_batches = math.ceil(num_examples / float(batch_size))

        # Setup the global step.
        slim.get_or_create_global_step()

        # How often to run the evaluation.
        eval_interval_secs = 10
        slim.evaluation.evaluation_loop(
            '',
            checkpoint_dir,
            log_dir,
            num_evals=num_batches,
            eval_op=names_to_updates.values(),
            summary_op=tf.summary.merge(summary_ops),
            eval_interval_secs=eval_interval_secs)
