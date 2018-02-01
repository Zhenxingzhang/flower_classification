from tensorflow.contrib import slim

import sys
sys.path.append("/data/slim/models/research/slim/")
from datasets import flowers
from preprocessing import inception_preprocessing

from src.data_preparation import dataset
from nets import inception

import tensorflow as tf
import os
import datetime


def train():
    # This might take a few minutes.
    TRAIN_SUMMARY_DIR = "/data/summary/flowers/train"
    l_rate = 0.001
    CHECKPOINT_DIR = '/data/checkpoints/flowers/'
    model_name = "slim_inception_v1"
    flowers_data_dir = "/data/flowers"
    batch_size = 64

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, model_name, str(l_rate))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print('Will save model to %s' % checkpoint_dir)

    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        image_size = inception.inception_v1.default_image_size

        train_dataset = flowers.get_split('train', flowers_data_dir)
        images, labels = dataset.load_batch(train_dataset, batch_size,
                                            height=image_size, width=image_size, is_training=False)

        tf.summary.image('images/train', images)

        # Create the model:
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v1(images, num_classes=5, is_training=True)

        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, 5)
        slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/total_loss', total_loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('prediction'):
                correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        train_summ_writer = tf.summary.FileWriter(
            os.path.join(TRAIN_SUMMARY_DIR, model_name, str(l_rate),
                         datetime.datetime.now().strftime("%Y%m%d-%H%M")), graph)

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=checkpoint_dir,
            number_of_steps=1000,  # For speed, we just do 1 epoch
            save_summaries_secs=1,
            summary_writer=train_summ_writer)

        print('Finished training. Final batch loss %d' % final_loss)


if __name__ == "__main__":
    train()
