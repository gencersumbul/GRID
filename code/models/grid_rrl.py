import tensorflow as tf
from utils import SummaryTracker
import networkx as nx
import numpy as np
from scipy import ndimage
from .grid import GRID
import numpy as np
   
class GRID_RRL(GRID):
    def __init__(self, config, summary_writer, **kwargs):
        super(GRID_RRL, self).__init__(config, summary_writer, **kwargs)
        self.neural_net = self.NeuralNet(self.config, self.input_shape)
        self.loss_tracker = SummaryTracker('reconstruction_loss', self.summary_writer)
   
    def find_edge_weight(self, sizes, center_of_label_masses, label_i_idx, label_j_idx, segmap):
        img_h, img_w = segmap.shape
        label_i_size_ratio = sizes[label_i_idx] / (img_h *img_w)
        label_j_size_ratio = sizes[label_j_idx] / (img_h *img_w)
        size_weight = (label_i_size_ratio + label_j_size_ratio) * (1 - (max(label_i_size_ratio, label_j_size_ratio)-0.5*(label_i_size_ratio+label_j_size_ratio)))
        distance_weight = 1 - (np.linalg.norm(np.array(center_of_label_masses[label_i_idx])-np.array(center_of_label_masses[label_j_idx])) / np.linalg.norm(np.array([img_h-1, img_w-1])))
        if (label_i_idx == label_j_idx) and (label_i_size_ratio > 0.5):
            return 1
        else:
            return (size_weight + distance_weight) /2. * (1 - (max(distance_weight, size_weight)-0.5*(distance_weight+size_weight))) #(size_weight + distance_weight) / 2.

    def task_loss(self, gt, out, sample_average=True):
        return self.reconstruction_loss(gt, out, sample_average)
    
    def construct_adjacency_matrix(self, segmap):
        segmap = segmap.numpy()
        labels, sizes = np.unique(segmap,return_counts=True)
        if -1 in labels:
            non_idx = np.where(labels == -1)[0][0]
            labels = np.delete(labels, non_idx)
            sizes = np.delete(sizes, non_idx)
        center_of_label_masses = [ndimage.measurements.center_of_mass(segmap == i) for i in labels]
    
        graph = nx.Graph()
        graph.add_nodes_from(range(self.config.nb_class))

        for label_i_idx, label_i in enumerate(labels):
            for label_j_idx, label_j in enumerate(labels):
                if (not graph.has_edge(label_i, label_j)):
                    graph.add_edge(label_i, label_j, weight = self.find_edge_weight(sizes, center_of_label_masses, label_i_idx, label_j_idx, segmap))
        adjacency = nx.adjacency_matrix(graph).todense()
        return adjacency[np.triu_indices(self.config.nb_class)] #, k = 1)]

    def prep_batch(self, batch, training):
        super().prep_batch(batch, training)
        self.batch['noisy_gt'] = np.reshape(np.array([self.construct_adjacency_matrix(segmap) for segmap in self.batch['noisy_seg_map']]), [-1, self.config.out_size])

    class NeuralNet(tf.keras.Model):
        def __init__(self, config, input_shape, **kwargs):
            super(GRID_RRL.NeuralNet, self).__init__(**kwargs)
            self.backbone = tf.keras.applications.DenseNet121(
                        weights=None, include_top=False, pooling='avg', input_shape=input_shape) 
            self.embed_dim = config.embed_dim

            self.latent = tf.keras.layers.Dense(self.embed_dim*2, activation=None)

            self.gen_decoder = tf.keras.layers.Dense(
                    units=1024,
                    activation=tf.keras.activations.relu
                )

            self.dis_adjacency_matrix_predictor = tf.keras.layers.Dense(
                units=config.out_size,
                activation=tf.keras.activations.relu
            )

            self.gen_adjacency_matrix_predictor = tf.keras.layers.Dense(
                units=config.out_size,
                activation=tf.keras.activations.relu
            )

        def call(self, inpts, training):
            out_dict = {}
            feature = self.backbone(inpts, training=training)
            out_dict['feature'] = feature

            with tf.name_scope('discriminative'):
                dis_adjacency_matrix_prediction = self.dis_adjacency_matrix_predictor(feature, training=training)
            
            out_dict['dis_out'] = dis_adjacency_matrix_prediction

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
                gen_adjacency_matrix_prediction = self.gen_adjacency_matrix_predictor(z_sample, training=training)

            out_dict['gen_out'] = gen_adjacency_matrix_prediction
            
            return out_dict