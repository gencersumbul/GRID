import tensorflow as tf
from utils import SummaryTracker
from .grid import GRID

class SoftmaxCrossEntropy(tf.keras.losses.Loss):
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__(reduction=tf.keras.losses.Reduction.NONE)
   
    @tf.function
    def __call__(self, labels, seg_logits, sample_average=True):
        pixel_ce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels, 
                        logits=seg_logits
                    )
        loss = tf.reduce_mean(tf.ragged.boolean_mask(pixel_ce,mask=pixel_ce != 0), axis=[1,2])
        return tf.reduce_mean(loss) if sample_average else loss
    
class GRID_PCE(GRID):
    def __init__(self, config, summary_writer, **kwargs):
        super(GRID_PCE, self).__init__(config, summary_writer, **kwargs)
        self.neural_net = self.NeuralNet(self.config, self.input_shape)
        self.loss_tracker = SummaryTracker('pce_loss', self.summary_writer)
        self.loss_fnc = SoftmaxCrossEntropy()

    def task_loss(self, gt, out, sample_average=True):
        return self.cross_entropy_loss(gt, out, sample_average)
     
    def cross_entropy_loss(self, label, logits, sample_average=True):
        return self.loss_fnc(label, logits, sample_average)
    
    def prep_batch(self, batch, training):
        super().prep_batch(batch, training)
        self.batch['noisy_gt'] = tf.one_hot(tf.cast(self.batch['noisy_seg_map'], tf.uint8), self.config.nb_class, axis=-1)
        self.batch['seg_map_encoded'] = tf.one_hot(tf.cast(self.batch['seg_map'], tf.uint8), self.config.nb_class, axis=-1)

    class NeuralNet(tf.keras.Model):
        def __init__(self, config, input_shape, **kwargs):
            super(GRID_PCE.NeuralNet, self).__init__(**kwargs)
            self.backbone = tf.keras.applications.DenseNet121(
                        weights=None, include_top=False, pooling=None, input_shape=input_shape) 
            self.embed_dim = config.embed_dim

            self.latent = tf.keras.layers.Conv2D(self.embed_dim*2, (1,1), activation=None)

            self.pooling = tf.keras.layers.GlobalAveragePooling2D()
            if config.dataset == 'DLRSD':
                self.dis_segPred_dec_a = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=(4,4), activation=tf.keras.activations.relu)
                self.dis_segPred_dec_b = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=6, strides=(4,4), activation=tf.keras.activations.relu)
                self.dis_segPred_dec_c = tf.keras.layers.Conv2DTranspose(filters=config.nb_class, kernel_size=30, strides=(2,2), activation=None)
            else:
                self.dis_segPred_dec_a = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2,2), activation=tf.keras.activations.relu)
                self.dis_segPred_dec_b = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=(4,4), activation=tf.keras.activations.relu)
                self.dis_segPred_dec_c = tf.keras.layers.Conv2DTranspose(filters=config.nb_class, kernel_size=8, strides=(4,4), activation=None)

            if config.dataset == 'DLRSD':
                self.gen_segPred_dec_a = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=(4,4), activation=tf.keras.activations.relu)
                self.gen_segPred_dec_b = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=6, strides=(4,4), activation=tf.keras.activations.relu)
                self.gen_segPred_dec_c = tf.keras.layers.Conv2DTranspose(filters=config.nb_class, kernel_size=30, strides=(2,2), activation=None)
            else:
                self.gen_segPred_dec_a = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2,2), activation=tf.keras.activations.relu)
                self.gen_segPred_dec_b = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=(4,4), activation=tf.keras.activations.relu)
                self.gen_segPred_dec_c = tf.keras.layers.Conv2DTranspose(filters=config.nb_class, kernel_size=8, strides=(4,4), activation=None)


            self.gen_decoder = tf.keras.layers.Conv2D(1024, (1,1), activation=tf.keras.activations.relu)
            
        def call(self, inpts, training):
            out_dict = {}
            feature = self.backbone(inpts, training=training)
            out_dict['feature_wo_pooling'] = feature
            out_dict['feature'] = self.pooling(feature)

            with tf.name_scope('discriminative'):
                out = self.dis_segPred_dec_a(feature)
                out = self.dis_segPred_dec_b(out)
                dis_logits = self.dis_segPred_dec_c(out)
            
            out_dict['dis_out'] = dis_logits

            with tf.name_scope('generative'):
                latent = self.latent(feature, training=training)
                out_dict['latent'] = latent
                mu, rho = tf.split(latent, num_or_size_splits=2, axis=-1)
                std = tf.math.log(1 + tf.math.exp(rho))
                out_dict['mu'] = mu
                out_dict['std'] = std
                z_sample = mu + std * tf.stop_gradient(tf.random.normal(shape=tf.shape(mu)))
                gen_feature = self.gen_decoder(z_sample, training=training)
                out_dict['gen_feature'] = gen_feature
                out = self.gen_segPred_dec_a(z_sample)
                out = self.gen_segPred_dec_b(out)
                gen_logits = self.gen_segPred_dec_c(out)

            out_dict['gen_out'] = gen_logits
            
            return out_dict