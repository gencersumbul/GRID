import json
import os
import numpy as np
from contextlib import contextmanager
from timeit import default_timer
from models import GRID_BCE, GRID_RRL, GRID_PCE
from datasets import DLRSD, BEN

CLASS_MAPPING = {
    'DLRSD': DLRSD,
    'BEN': BEN,
    'GRID-BCE': GRID_BCE,
    'GRID-PCE': GRID_PCE,
    'GRID-RRL': GRID_RRL
}

def model_selector(config, summary_writer):
    return CLASS_MAPPING[config.model_name](config, summary_writer) 

def get_shuffle_buffer_size(dataset, is_training=True):
    if is_training:
        if dataset == 'BEN':
            return 10000 #39000
        elif dataset == 'DLRSD': 
            return 1680
    else:
        return 0

def save_config(config, training_time):
    exp_ids = []
    configs_path = config.dumps.configs
    for config_f in os.listdir(configs_path):
        if '.json' in config_f:
            with open(os.path.join(configs_path, config_f), 'r') as fp:
                contents = json.load(fp)
            exp_ids.append(contents['exp_id'])

    if len(exp_ids) == 0:
        config = config._replace(exp_id = 0)
    elif len(np.where(np.array(exp_ids) == config.exp_id)[0]) > 0:
        config = config._replace(exp_id = int(max(exp_ids) + 1))

    config = config._replace(training_time = '{:0.1f}'.format(training_time))
    save_file_name = os.path.join(
        configs_path, config.suffix + '.json')
    
    with open(save_file_name, 'w') as fp:
        res = dict(config._asdict())
        res['dumps'] = dict(config.dumps._asdict())   
        json.dump(res, fp)
    return save_file_name

def select_gpu(gpu_number):
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_number],'GPU')
        except RuntimeError as e:
            print(e)
        
@contextmanager
def timer_calc():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start