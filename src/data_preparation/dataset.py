from tensorflow.contrib import slim

import sys
sys.path.append("/data/slim/models/research/slim/")
from datasets import flowers
from preprocessing import inception_preprocessing

import tensorflow as tf


num_samples = 3320
num_classes = 5


def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False, epochs=None):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
      epochs : how many number of epochs, default value: None.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=24 + 3 * batch_size,
        num_epochs=epochs, common_queue_min=24)
    image_raw, label = data_provider.get(['image', 'label'])

    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)

    # Batch it up.
    images_, labels_ = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=4 * batch_size,
        allow_smaller_final_batch=True)

    return images_, labels_


if __name__ == '__main__':
    flowers_data_dir = "/data/flowers"
    train_dataset = flowers.get_split('validation', flowers_data_dir)
    images, labels = load_batch(train_dataset, 200, is_training=False, epochs=1)

    total_train_samples = 0

    print(images)
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        with slim.queues.QueueRunners(sess):
            try:
                for step in range(1000):
                    imgs, y_ = sess.run([images, labels])
                    total_train_samples += imgs.shape[0]
                    print(imgs.shape[0])
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached, {} total training example!'.format(total_train_samples))
