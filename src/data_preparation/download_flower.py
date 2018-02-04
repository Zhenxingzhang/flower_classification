import tensorflow as tf
import sys
sys.path.append("/data/slim/models/research/slim/")
from datasets import dataset_utils

if __name__ == '__main__':
    url = "http://download.tensorflow.org/data/flowers.tar.gz"
    flowers_data_dir = '/data/flowers'

    if not tf.gfile.Exists(flowers_data_dir):
        tf.gfile.MakeDirs(flowers_data_dir)

    # dataset_utils.download_and_uncompress_tarball(url, flowers_data_dir)

    url = "http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz"
    checkpoints_dir = '/data/inception_resnet/v2'

    if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)

    dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)
