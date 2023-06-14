import tensorflow as tf

BAND_STATS = {
            'mean': {
                'B01': 340.76769064,
                'B02': 429.9430203,
                'B03': 614.21682446,
                'B04': 590.23569706,
                'B05': 950.68368468,
                'B06': 1792.46290469,
                'B07': 2075.46795189,
                'B08': 2218.94553375,
                'B8A': 2266.46036911,
                'B09': 2246.0605464,
                'B11': 1594.42694882,
                'B12': 1009.32729131
            },
            'std': {
                'B01': 554.81258967,
                'B02': 572.41639287,
                'B03': 582.87945694,
                'B04': 675.88746967,
                'B05': 729.89827633,
                'B06': 1096.01480586,
                'B07': 1273.45393088,
                'B08': 1365.45589904,
                'B8A': 1356.13789355,
                'B09': 1302.3292881,
                'B11': 1079.19066363,
                'B12': 818.86747235
            }
        }

class BEN:
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
    
    @tf.function
    def normalize_band(self, band, name):
        return (band - BAND_STATS['mean'][name]) / BAND_STATS['std'][name]

    def parse_function(self, example_proto):
        parsed_features = tf.io.parse_single_example(
                example_proto, 
                {
                    'B01': tf.io.FixedLenFeature([20*20], tf.float32),
                    'B02': tf.io.FixedLenFeature([120*120], tf.float32),
                    'B03': tf.io.FixedLenFeature([120*120], tf.float32),
                    'B04': tf.io.FixedLenFeature([120*120], tf.float32),
                    'B05': tf.io.FixedLenFeature([60*60], tf.float32),
                    'B06': tf.io.FixedLenFeature([60*60], tf.float32),
                    'B07': tf.io.FixedLenFeature([60*60], tf.float32),
                    'B08': tf.io.FixedLenFeature([120*120], tf.float32),
                    'B8A': tf.io.FixedLenFeature([60*60], tf.float32),
                    'B09': tf.io.FixedLenFeature([20*20], tf.float32),
                    'B11': tf.io.FixedLenFeature([60*60], tf.float32),
                    'B12': tf.io.FixedLenFeature([60*60], tf.float32),
                    'BEN-19_labels': tf.io.VarLenFeature(dtype=tf.string),
                    'BEN-19_multi_hot': tf.io.FixedLenFeature([19], tf.float32),
                    'noisy_BEN-19_multi_hot': tf.io.FixedLenFeature([19], tf.float32),
                    'patch_name': tf.io.VarLenFeature(dtype=tf.string)
                }
            )

        out = {
            'patch_name': parsed_features['patch_name'], 
            'BEN-19_labels': parsed_features['BEN-19_labels'], 
            'BEN-19_multi_hot': parsed_features['BEN-19_multi_hot'],
            'noisy_BEN-19_multi_hot': parsed_features['noisy_BEN-19_multi_hot']
        }

        for name, shape in zip(
                ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'] , 
                [[20, 20] , [120, 120], [120, 120], [120, 120], [60, 60], [60, 60], [60, 60], [120, 120], [60, 60], [20, 20], [60, 60], [60, 60]]):
            out[name] = tf.reshape(self.normalize_band(parsed_features[name], name), shape)
        return out