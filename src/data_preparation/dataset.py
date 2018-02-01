from tensorflow.contrib import slim

import sys
sys.path.append("/data/slim/models/research/slim/")
from datasets import flowers
from preprocessing import inception_preprocessing

import tensorflow as tf


def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])

    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)


    # Batch it up.
    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=2,
        capacity=10 * batch_size,
        min_after_dequeue=batch_size)

    return images, labels


if __name__ == '__main__':
    flowers_data_dir = "/data/flowers"
    train_dataset = flowers.get_split('train', flowers_data_dir)
    images, labels = load_batch(train_dataset, 10)

    print(images)
    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            imgs, y_ = sess.run([images, labels])
            print(imgs.shape)
            print(y_)
