import os
import sys
import datetime
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from src.data_preparation import dataset

sys.path.append("/data/slim/models/research/slim/")
from datasets import flowers

if __name__ == '__main__':
    slim = tf.contrib.slim

    model_name = "slim_inception_v1"
    l_rate = 0.001
    CHECKPOINT_DIR = '/data/checkpoints/flowers/'
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, model_name, str(l_rate))

    # VAL_SUMMARY_DIR = "/data/summary/flowers/val"
    # log_dir = os.path.join(VAL_SUMMARY_DIR, model_name, str(l_rate), datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    flowers_data_dir = "/data/flowers"
    batch_size = 10

    image_size = inception.inception_v1.default_image_size
    val_dataset = flowers.get_split('validation', flowers_data_dir)

    # Load the data
    images, labels = dataset.load_batch(
        val_dataset, batch_size, height=image_size, width=image_size, is_training=False)

    # Create the model:
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(images, num_classes=5, is_training=False)
        predictions = tf.argmax(logits, 1)

    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    init_fn = slim.assign_from_checkpoint_fn(
      checkpoint_path,
      slim.get_variables_to_restore())

    # Choose the metrics to compute:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        "eval/accuracy": slim.metrics.streaming_accuracy(predictions, labels),
        'eval/precision': slim.metrics.streaming_precision(predictions, labels),
    })

    # Evaluate the model using 1000 batches of data:
    num_batches = 10

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            sess.run(tf.local_variables_initializer())
            init_fn(sess)

            print("Evaluation started....")
            for batch_id in range(num_batches):
                print("evaluate batch {}".format(batch_id))
                sess.run(names_to_updates.values())

            metric_values = sess.run(names_to_values.values())
            for metric, value in zip(names_to_values.keys(), metric_values):
                print('Metric %s has value: %f' % (metric, value))
            # accu_ = sess.run(value_op)
            # print("accuracy: {}".format(accu_))

