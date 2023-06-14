import tensorflow as tf

# https://sites.google.com/view/zhouwx/dataset#h.p_hQS2jYeaFpV0

BAND_STATS = {
            'mean': [76.54768, 76.80208, 73.49884],
            'std': [72.219376, 70.419426, 63.12712]
        }

class DLRSD:
    def __init__(self, tfrecord_paths, batch_size, shuffle_buffer_size):
        dataset = tf.data.TFRecordDataset(tfrecord_paths, buffer_size=int(1e+6),num_parallel_reads=tf.data.experimental.AUTOTUNE)
        if shuffle_buffer_size > 0:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.map(
            self.parse_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False
        )
        dataset = dataset.batch(batch_size, drop_remainder=False)
        self.dataset = dataset.prefetch(10)
    
    def normalize_img(self, img):
        return tf.stack(
                    [
                        (img[:,:,0] - BAND_STATS['mean'][0]) / BAND_STATS['std'][0],
                        (img[:,:,1] - BAND_STATS['mean'][1]) / BAND_STATS['std'][1],
                        (img[:,:,2] - BAND_STATS['mean'][2]) / BAND_STATS['std'][2]
                    ], 
                    axis=-1
                ) 

    def parse_function(self, example_proto):
        parsed_features = tf.io.parse_single_example(
                example_proto, 
                {
                    'label': tf.io.FixedLenFeature([17], tf.float32),
                    'noisy_label': tf.io.FixedLenFeature([17], tf.float32),
                    'img': tf.io.FixedLenFeature([256*256*3], tf.float32),
                    'seg_map': tf.io.FixedLenFeature([256*256], tf.float32),
                    'noisy_seg_map': tf.io.FixedLenFeature([256*256], tf.float32),
                    'patch_name': tf.io.VarLenFeature(dtype=tf.string)
                }
            )
        img = tf.reshape(parsed_features['img'], [256, 256, 3])
        seg_map = tf.reshape(parsed_features['seg_map'], [256, 256])
        noisy_seg_map = tf.reshape(parsed_features['noisy_seg_map'], [256, 256])
        return {
            'img': self.normalize_img(img),
            'seg_map': seg_map,
            'noisy_seg_map': noisy_seg_map,
            'label': parsed_features['label'],
            'noisy_label': parsed_features['noisy_label'],
            'patch_name': parsed_features['patch_name']
        }