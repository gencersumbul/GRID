import tensorflow as tf
import os

class GRID():
    def __init__(self, config, summary_writer):
        self.config = config
        self.summary_writer = summary_writer 
        self.optimizer = tf.optimizers.get({
            'class_name': self.config.optimizer, 
            'config': {
                'learning_rate': self.config.learning_rate
            }
        })
        self.batch = None
        self.input_shape = [120, 120, 10] if self.config.dataset == 'BEN' else [224, 224, 3]
        self.noisy_sel_ratio = self.config.noisy_sel_prcnt * 0.01

    def finish_training(self, save_last_state):
        if save_last_state:
            self.neural_net.save(os.path.join(self.config.dumps.model_weights, self.config.suffix))
        self.summary_writer.close()

    def reset_trackers(self):
        self.loss_tracker.reset_states()

    def train_loss_val(self):
        return self.loss_tracker.result()

    def normalize(self, tensor):
            return tf.math.divide_no_nan(
                tf.subtract(
                    tensor, 
                    tf.reduce_min(tensor)
                ), 
                tf.subtract(
                    tf.reduce_max(tensor), 
                    tf.reduce_min(tensor)
                )
            )

    def forward_pass(self, batch, training):
        return self.neural_net(batch['img'], training)

    def training_step(self, batch, kl_weight=1.):
        with tf.GradientTape(persistent=True) as tape:
            out = self.forward_pass(batch, training=True)
            dis_loss, gen_loss, latent_reconstruction_loss, kl_divergence, gen_dis_loss = self.loss(batch, out)
            gen_loss = gen_loss + latent_reconstruction_loss + kl_weight * kl_divergence
        tvars_dict = self.get_tvars()
        dis_gradients = tape.gradient(dis_loss, tvars_dict['discriminative'])
        gen_gradients = tape.gradient(gen_loss, tvars_dict['generative'])
        bacbone_gradients = tape.gradient(gen_dis_loss, tvars_dict['backbone'])
        del tape
        self.optimizer.apply_gradients(zip(dis_gradients, tvars_dict['discriminative']))
        self.optimizer.apply_gradients(zip(gen_gradients, tvars_dict['generative']))
        self.optimizer.apply_gradients(zip(bacbone_gradients, tvars_dict['backbone']))
        self.loss_tracker(dis_loss)

    def reconstruction_loss(self, gt, prediction, sample_average=True):
        loss = tf.reduce_mean(tf.square(gt - prediction), axis=-1)
        return tf.reduce_mean(loss) if sample_average else loss
    
    def loss(self, batch, out):
        batch_size = batch['img'].shape.as_list()[0]
        mu = out['mu']
        std = out['std']

        kl_divergence = - 0.5 * tf.math.reduce_sum(
            1 + tf.math.log(tf.math.square(std)+tf.keras.backend.epsilon()) - tf.math.square(mu) - tf.math.square(std),
            axis=1)
        kl_divergence = tf.reduce_mean(kl_divergence)

        feat_key = 'feature_wo_pooling' if 'PCE' in self.config.model_name else 'feature'
        latent_reconstruction_loss = self.reconstruction_loss(tf.stop_gradient(out[feat_key]), out['gen_feature'])
        gen_loss =  self.task_loss(batch['noisy_gt'], out['gen_out'], sample_average=False)
        dis_loss =  self.task_loss(batch['noisy_gt'], out['dis_out'], sample_average=False)

        loss_diff = self.normalize(dis_loss) - self.normalize(gen_loss)
        _, noisy_indices = tf.nn.top_k(loss_diff, k=int(batch_size * self.noisy_sel_ratio))
        _, safe_indices = tf.nn.top_k(tf.negative(loss_diff), k=batch_size - int(batch_size * self.noisy_sel_ratio))

        gen_dis_loss = (tf.reduce_sum(tf.gather(gen_loss, noisy_indices)) + tf.reduce_sum(tf.gather(dis_loss, safe_indices))) / float(batch_size)
        
        gen_loss = tf.reduce_mean(gen_loss)
        dis_loss = tf.reduce_mean(dis_loss)
        return dis_loss, gen_loss, latent_reconstruction_loss, kl_divergence, gen_dis_loss 
    
    def get_tvars(self):
        tvars = self.neural_net.trainable_variables
        tvars_dict = {**{'backbone':[]}, **{'discriminative':[]}, **{'generative':[]}}
        for var_idx, var in enumerate(tvars):
            if 'discriminative' in var.name:
                tvars_dict['discriminative'].append(var)
            elif 'generative' in var.name:
                tvars_dict['generative'].append(var)
            else:
                tvars_dict['backbone'].append(var)
        assert len(tvars_dict['backbone']) != 0
        assert len(tvars_dict['generative']) != 0
        assert len(tvars_dict['discriminative']) != 0
        return tvars_dict
    
    def get_features(self, batch):
        return self.neural_net(batch['img'], training=False)['feature']
        
    def prep_batch(self, batch, training):
        def img_process(img):
            if training:
                if self.config.dataset == 'DLRSD':
                    img = tf.keras.layers.experimental.preprocessing.RandomCrop(224, 224, seed=42)(img, training=True)
                img = tf.image.random_flip_left_right(img, seed=42)
                img = tf.image.random_flip_up_down(img, seed=42) 
                img = tf.image.random_brightness(img, max_delta=0.2, seed=42)
            elif self.config.dataset == 'DLRSD':
                img = tf.keras.layers.experimental.preprocessing.CenterCrop(224, 224)(img)
            return img
        
        if self.config.dataset == 'DLRSD':  
            self.batch = {
                'img': img_process(batch['img']),
                'label': batch['label'],
                'noisy_label': batch['noisy_label'],
                'seg_map': batch['seg_map'],
                'noisy_seg_map': batch['noisy_seg_map'],
                'patch_name': tf.sparse.to_dense(batch['patch_name'], default_value='x')
            }
        else:
            img = tf.concat(
                    [
                        tf.stack([batch['B04'], batch['B03'], batch['B02'], batch['B08']], axis=3),
                        tf.image.resize(
                            tf.stack(
                                [batch['B05'], batch['B06'], batch['B07'], batch['B8A'], batch['B11'], batch['B12']], 
                                axis=3
                            ), 
                            [120, 120],
                            method=tf.image.ResizeMethod.BICUBIC
                        )
                    ], 
                    axis=3
                )

            self.batch = {
                'img': img_process(img),
                'label': batch['BEN-19_multi_hot'],
                'noisy_label': batch['noisy_BEN-19_multi_hot'],
                'patch_name': tf.sparse.to_dense(batch['patch_name'], default_value='x')
            }
    
    def training_epoch(self, train_dataset, epoch):
        self.nb_train_samples = 0
        for batch_id, batch in enumerate(train_dataset):
            self.prep_batch(batch, training=True)
            self.nb_train_samples += len(self.batch['img'])
            self.training_step(self.batch)