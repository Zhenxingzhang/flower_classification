import yaml
from dictionaries import Dict


def parse_config_file(config_filename):
    with open(config_filename, 'r') as yml_file:
        cfg = yaml.load(yml_file)

    arg_coonfig = Dict(copy=True, name='config')
    arg_coonfig.MODEL_NAME = str(cfg["MODEL"]["MODEL_NAME"])
    arg_coonfig.PRETAIN_MODEL = str(cfg["MODEL"]["PRETRAIN_MODEL"])
    arg_coonfig.PRETAIN_MODEL_PATH = str(cfg["MODEL"]["PRETAIN_MODEL_PATH"])
    arg_coonfig.PRETAIN_MODEL_URL = str(cfg["MODEL"]["PRETAIN_MODEL_URL"])
    arg_coonfig.INPUT_HEIGHT = int(cfg["MODEL"]["INPUT_HEIGHT"])
    arg_coonfig.INPUT_WIDTH = int(cfg["MODEL"]["INPUT_WIDTH"])
    arg_coonfig.CATEGORIES = int(cfg["MODEL"]["CLASSES"])

    arg_coonfig.TRAIN_BATCH_SIZE = int(cfg["TRAIN"]["BATCH_SIZE"])
    arg_coonfig.TRAIN_EPOCHS_COUNT = int(cfg["TRAIN"]["EPOCHS_COUNT"])
    arg_coonfig.TRAIN_LEARNING_RATE = float(cfg["TRAIN"]["LEARNING_RATE"])
    arg_coonfig.TRAIN_KEEP_PROB = float(cfg['TRAIN']['KEEP_PROB'])
    arg_coonfig.TRAIN_TF_RECORDS = str(cfg["TRAIN"]["TF_RECORDS_PATH"])
    arg_coonfig.TRAIN_EPOCHS_BEFORE_DECAY = float(cfg["TRAIN"]["TRAIN_EPOCHS_BEFORE_DECAY"])
    arg_coonfig.TRAIN_RATE_DECAY_FACTOR = float(cfg["TRAIN"]["TRAIN_RATE_DECAY_FACTOR"])

    arg_coonfig.EVAL_BATCH_SIZE = cfg["EVAL"]["BATCH_SIZE"]
    arg_coonfig.EVAL_TF_RECORDS = str(cfg["EVAL"]["TF_RECORDS"])

    return arg_coonfig
