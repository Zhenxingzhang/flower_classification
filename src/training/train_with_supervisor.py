from tensorflow.contrib import slim

import sys
sys.path.append("/data/slim/models/research/slim/")
from datasets import flowers
from preprocessing import inception_preprocessing

from src.data_preparation import dataset
from nets import inception_resnet_v2

import tensorflow as tf
import os
import time
import logging


def run():

    # Specify where the Model, trained on ImageNet, was saved.
    fine_tune_model_dir = "/data/inception_resnet/v2/inception_resnet_v2_2016_08_30.ckpt"

    model_name = "inception_resnet_v2_ft"

    # This might take a few minutes.
    TRAIN_SUMMARY_DIR = os.path.join("/data/summary/flowers/", model_name, "train")
    initial_learning_rate = 0.0001
    learning_rate_decay_factor = 0.7
    CHECKPOINT_DIR = '/data/checkpoints/flowers/'
    flowers_data_dir = "/data/flowers"
    batch_size = 32
    num_epochs = 10

    num_epochs_before_decay = 2

    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(fine_tune_model_dir):
        os.mkdir(fine_tune_model_dir)

    # ======================= TRAINING PROCESS =========================
    # Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # First create the dataset and load one batch
        train_dataset = flowers.get_split('train', flowers_data_dir)
        images, labels = dataset.load_batch(train_dataset, batch_size=batch_size)

        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        # Create the model inference
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, end_points = \
                inception_resnet_v2.inception_resnet_v2(images, num_classes=dataset.num_classes, is_training=True)

        # Define the scopes that you want to exclude for restoration
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        # Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

        # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

        # Create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        # Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        # Now we need to create a training step function that runs both the train_op,
        # metrics_op and updates the global_step concurrently.
        def train_step(sess_, train_op_, global_step_):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            # Check the time for each sess run
            start_time = time.time()
            total_loss_, global_step_count, _ = sess_.run([train_op_, global_step_, metrics_op])
            time_elapsed = time.time() - start_time

            # Run the logging to print some results
            print('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss_, global_step_count

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, fine_tune_model_dir)

        # Define your supervisor for running a managed session.
        # Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=TRAIN_SUMMARY_DIR, summary_op=None, init_fn=restore_fn)

        # Run the managed session
        with sv.managed_session() as sess:
            for step in xrange(num_steps_per_epoch * num_epochs):
                # At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    print('Epoch %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    print('Current Learning Rate: %s', learning_rate_value)
                    print('Current Streaming Accuracy: %s', accuracy_value)

                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                        [logits, probabilities, predictions, labels])
                    # print 'logits: \n', logits_value
                    # print 'Probabilities: \n', probabilities_value
                    print 'predictions: \n', predictions_value
                    print 'Labels:\n', labels_value

                # Log the summaries every 10 step.
                if step % 10 == 0:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                # If not, simply run the training step
                else:
                    loss, _ = train_step(sess, train_op, sv.global_step)

            # We log the final training loss and accuracy
            print('Final Loss: %s', loss)
            print('Final Accuracy: %s', sess.run(accuracy))

            # Once all the training has been done, save the log files and checkpoint model
            print('Finished training! Saving model to disk now.')
            # saver.save(sess, "./flowers_model.ckpt")
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)


if __name__ == '__main__':
    run()
