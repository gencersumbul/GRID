import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from utils import Config, save_config, select_gpu, timer_calc, model_selector, CLASS_MAPPING, get_shuffle_buffer_size
import json

class Trainer():
    def __init__(self, config):
        self.config = config
        self.summary_writer = tf.summary.create_file_writer(os.path.join(self.config.dumps.summaries, self.config.suffix))
        self.epoch = 0

    def prep_training(self):
        self.model = model_selector(self.config, self.summary_writer)
        self.train_dataset = CLASS_MAPPING[self.config.dataset](
            self.config.train_tfrecord_paths, self.config.batch_size, get_shuffle_buffer_size(self.config.dataset, is_training=True)).dataset
        self.val_dataset = CLASS_MAPPING[self.config.dataset](
            self.config.val_tfrecord_paths, self.config.batch_size, get_shuffle_buffer_size(self.config.dataset, is_training=False)).dataset
        self.training_time = 0.

    def training_loop(self):
        self.prep_training()
        save_last_state = True
        print('training is started')
        for epoch in range(self.config.nb_epoch):
            with timer_calc() as elapsed_time:
                self.model.training_epoch(self.train_dataset, epoch)
                epoch_time = elapsed_time()
                print('epoch {} is finished with {} samples within {:0.2f} seconds, loss {}'.format(epoch + 1, self.model.nb_train_samples, epoch_time, self.model.train_loss_val()))  
                self.training_time += epoch_time
            self.epoch = epoch + 1
            self.model.reset_trackers()
        self.model.finish_training(save_last_state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= 'Training script')
    parser.add_argument('config', nargs='?', default='', help= 'json config file')
    parser_args = parser.parse_args()
    
    with open(parser_args.config, 'r') as fp:
        config_dict = json.load(fp)
    config = Config(config_dict, training=True)
    if (config.dataset == 'BEN') and (config.model_name == 'GRID-PCE' or config.model_name == 'GRID-RRL'):
        print('BEN can be used with GRID-BCE')
    else:
        select_gpu(config.gpu)
        try:
            trainer = Trainer(config)
            trainer.training_loop()
        finally:
            if trainer.epoch > 0:
                trainer.model.finish_training(save_last_state=True)
                save_config(config, trainer.training_time)
