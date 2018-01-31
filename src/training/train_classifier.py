from tensorflow.contrib import slim

import sys
sys.path.append("/data/slim/models/research/slim/")
from datasets import flowers
from preprocessing import inception_preprocessing

from src.data_preparation import dataset
from nets import inception

import tensorflow as tf


def train():
    # This might take a few minutes.
    train_dir = '/data/checkpoints/flower/'
    flowers_data_dir = "/data/flowers"
    batch_size = 64

    print('Will save model to %s' % train_dir)

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        image_size = inception.inception_v1.default_image_size

        train_dataset = flowers.get_split('train', flowers_data_dir)
        images, labels = dataset.load_batch(train_dataset, batch_size,
                                            height=image_size, width=image_size, is_training=True)

        tf.summary.image('images', images)

        # Create the model:
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v1(images, num_classes=5, is_training=True)

        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, 5)
        slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/train_Loss', total_loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('prediction'):
                correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('train_accuracy', accuracy)

        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=train_dir,
            number_of_steps=1000,  # For speed, we just do 1 epoch
            save_summaries_secs=1)

        print('Finished training. Final batch loss %d' % final_loss)


if __name__ == "__main__":
    train()
