"""Defines config tuple"""

from collections import namedtuple
import json
from datetime import datetime

class Config():
    """Helper class to build config object"""
    def __new__(cls, contents, training):
        DEFAULTS = {
            "exp_id": 0,
            "training_time": '',
            "time": '',
            "suffix" : "",
            "gpu": 0,
            "dumps": {
                "model_weights": "../dumps/weights/",
                "summaries": "../dumps/summaries/",
                "configs": "../dumps/configs/",
                "features": "../dumps/features/"
            },
            "dataset": "DLRSD",
            "embed_dim": 128,
            "model_name": "GRID-BCE",
            "noise_prcnt": 0,
            "noisy_sel_prcnt": 10,
            "batch_size": 128,
            "train_tfrecord_paths": ["../dumps/data/DLRSD/TFRecord/train.tfrecord"],
            "test_tfrecord_paths": ["../dumps/data/DLRSD/TFRecord/test.tfrecord"],
            "val_tfrecord_paths": ["../dumps/data/DLRSD/TFRecord/val.tfrecord"],
            "nb_epoch":100,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "out_size": 0,
            "nb_class": 17
        }
        for key in list(contents):
            if type(contents[key]) == type({}):
                for sub_key in contents[key].keys():
                    if not sub_key in DEFAULTS[key].keys():
                        DEFAULTS[key][sub_key] = contents[key][sub_key]
            else:
                if not key in DEFAULTS.keys():
                    DEFAULTS[key] = contents[key]

        for key in list(DEFAULTS):
            if type(DEFAULTS[key]) == type({}):
                for sub_key in DEFAULTS[key].keys():
                    if key in contents.keys():
                        if not sub_key in contents[key].keys():
                            contents[key][sub_key] = DEFAULTS[key][sub_key]
                    else:
                        contents[key] = {}
                        contents[key][sub_key] = DEFAULTS[key][sub_key]
            else:
                if not key in contents.keys():
                    contents[key] = DEFAULTS[key]

        for key in DEFAULTS.keys():
            if type(DEFAULTS[key]) == type({}):
                temp = namedtuple(key, DEFAULTS[key].keys())
                DEFAULTS[key] = temp(**DEFAULTS[key])

        config = namedtuple('config', list(DEFAULTS.keys()))
        config_tuple = config(**DEFAULTS)

        config_tuple = config_tuple._replace(**contents)
        
        temp = namedtuple('dumps', contents['dumps'].keys())
        config_tuple = config_tuple._replace(dumps = temp(**contents['dumps']))
       
        config_tuple = config_tuple._replace(nb_class=17 if config_tuple.dataset == "DLRSD" else 19)
        out_size = int(0.5*config_tuple.nb_class*(config_tuple.nb_class-1) + config_tuple.nb_class)
        config_tuple = config_tuple._replace(out_size = out_size)

        if config_tuple.dataset == 'BEN':
            config_tuple = config_tuple._replace(train_tfrecord_paths = ["../dumps/data/BEN/TFRecord/train.tfrecord"])
            config_tuple = config_tuple._replace(val_tfrecord_paths = ["../dumps/data/BEN/TFRecord/val.tfrecord"])
            config_tuple = config_tuple._replace(test_tfrecord_paths = ["../dumps/data/BEN/TFRecord/test.tfrecord"])
            config_tuple = config_tuple._replace(nb_class = 19)

        if training:
            m_name = '-'.join([
                config_tuple.dataset, 
                config_tuple.model_name, 
                'embed' + str(config_tuple.embed_dim),
                'noise_prcnt' + str(config_tuple.noise_prcnt),
                'bs' + str(config_tuple.batch_size), 
                'epoch' + str(config_tuple.nb_epoch),
                'noisy_sel_prcnt' + str(config_tuple.noisy_sel_prcnt)])
            if not config_tuple.noise_prcnt == 0:
                tr_path = config_tuple.train_tfrecord_paths[0].split('.tfrecord')[0] + '_noise_prcnt' + str(config_tuple.noise_prcnt) + '.tfrecord'
                config_tuple = config_tuple._replace(train_tfrecord_paths=[tr_path])

            if not config_tuple.suffix == '':
                m_name += '-' + config_tuple.suffix
            now = datetime.now()
            exp_time = now.strftime("%d-%m-%Y-%H-%M")
            config_tuple = config_tuple._replace(suffix=m_name)
            config_tuple = config_tuple._replace(time = exp_time)
        return config_tuple

