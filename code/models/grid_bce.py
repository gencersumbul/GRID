import tensorflow as tf
from utils import SummaryTracker
from .grid import GRID

class GRID_BCE(GRID):
    def __init__(self, config, summary_writer, **kwargs):
        super(GRID_BCE, self).__init__(config, summary_writer, **kwargs)
        self.neural_net = self.NeuralNet(self.config, self.input_shape)
        self.loss_tracker = SummaryTracker('bce_loss', self.summary_writer)

    def task_loss(self, gt, out, sample_average=True):
        return self.cross_entropy_loss(gt, out, sample_average)
    
    def cross_entropy_loss(self, label, logits, sample_average=True):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits),axis=-1)
        return tf.reduce_mean(loss) if sample_average else loss

    def prep_batch(self, batch, training):
        super().prep_batch(batch, training)
        self.batch['noisy_gt'] = self.batch['noisy_label']

    class NeuralNet(tf.keras.Model):
        def __init__(self, config, input_shape, **kwargs):
            super(GRID_BCE.NeuralNet, self).__init__(**kwargs)
            self.backbone = tf.keras.applications.DenseNet121(
                        weights=None, include_top=False, pooling='avg', input_shape=input_shape) 
            self.embed_dim = config.embed_dim

            self.latent = tf.keras.layers.Dense(self.embed_dim*2, activation=None)

            self.gen_decoder = tf.keras.layers.Dense(
                    units=1024,
                    activation=tf.keras.activations.relu
                )

            self.dis_clsPred_head = tf.keras.layers.Dense(
                units=config.nb_class,
                activation=None
            )

            self.gen_clsPred_head = tf.keras.layers.Dense(
                units=config.nb_class,
                activation=None
            )

        def call(self, inpts, training):
            out_dict = {}
            feature = self.backbone(inpts, training=training)
            out_dict['feature'] = feature

            with tf.name_scope('discriminative'):
                dis_logits = self.dis_clsPred_head(feature, training=training)
            
            out_dict['dis_out'] = dis_logits

            with tf.name_scope('generative'):
                latent = self.latent(feature, training=training)
                out_dict['latent'] = latent
                mu, rho = tf.split(latent, num_or_size_splits=2, axis=1)
                std = tf.math.log(1 + tf.math.exp(rho))
                out_dict['mu'] = mu
                out_dict['std'] = std
                z_sample = mu + std * tf.stop_gradient(tf.random.normal(shape=(int(self.embed_dim),)))
                gen_feature = self.gen_decoder(z_sample, training=training)
                out_dict['gen_feature'] = gen_feature
                gen_logits = self.gen_clsPred_head(z_sample, training=training)

            out_dict['gen_out'] = gen_logits
            
            return out_dict