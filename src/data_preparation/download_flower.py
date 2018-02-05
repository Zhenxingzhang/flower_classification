import tensorflow as tf
import sys
sys.path.append("/data/slim/models/research/slim/")
from datasets import dataset_utils
from src.utils import helper

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c',
                        dest="config_filename", type=str, required=True,
                        help='the config file name must be provide')
    args = parser.parse_args()

    arg_config = helper.parse_config_file(args.config_filename)

    url = "http://download.tensorflow.org/data/flowers.tar.gz"
    flowers_data_dir = '/data/flowers'

    if not tf.gfile.Exists(flowers_data_dir):
        tf.gfile.MakeDirs(flowers_data_dir)

    # dataset_utils.download_and_uncompress_tarball(url, flowers_data_dir)

    checkpoints_dir = os.path.join('/data/pretain_model', arg_config.PRETAIN_MODEL)

    if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)

    dataset_utils.download_and_uncompress_tarball(arg_config.PRETAIN_MODEL_URL, checkpoints_dir)

