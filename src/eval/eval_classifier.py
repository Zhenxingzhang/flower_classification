import tensorflow as tf

import sys, os
sys.path.append("/data/slim/models/research/slim/")
from datasets import flowers
from preprocessing import inception_preprocessing
from nets import inception

from src.data_preparation import dataset
import math

if __name__ == '__main__':
    slim = tf.contrib.slim

    checkpoint_dir = '/data/checkpoints/flower/'
    log_dir = os.path.join(checkpoint_dir, "eval")
    flowers_data_dir = "/data/flowers"
    batch_size = 64

    image_size = inception.inception_v1.default_image_size

    # This might take a few minutes.
    with tf.Graph().as_default():

        # Load the data
        train_dataset = flowers.get_split('validation', flowers_data_dir)
        images, labels = dataset.load_batch(train_dataset, batch_size,
                                            height=image_size, width=image_size, is_training=False)

        tf.summary.image('images', images)

        # Create the model:
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v1(images, num_classes=5, is_training=False)
            predictions = tf.argmax(logits, 1)

        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy/eval_accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'eval/Recall@1': slim.metrics.streaming_recall_at_k(logits, labels, 1),
            #'eval/precision': slim.metrics.precision(predictions, labels),
            # 'eval/recall': slim.metrics.recall(mean_relative_errors, 0.3),
        })

        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        num_examples = 1000
        num_batches = math.ceil(num_examples / float(batch_size))

        # Setup the global step.
        slim.get_or_create_global_step()

        # How often to run the evaluation.
        eval_interval_secs = 60
        slim.evaluation.evaluation_loop(
            '',
            checkpoint_dir,
            log_dir,
            num_evals=num_batches,
            eval_op=names_to_updates.values(),
            summary_op=tf.summary.merge(summary_ops),
            eval_interval_secs=eval_interval_secs)
