import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from utils import Config, select_gpu, CLASS_MAPPING, timer_calc, model_selector, get_shuffle_buffer_size
from tqdm import tqdm
import h5py
import ray
import json

class Retrieval():
    def __init__(self, config):
        self.config = config
        self.summary_writer = tf.summary.create_noop_writer() #create_file_writer(os.path.join(self.config.dumps.summaries, self.config.suffix))
        self.query_feat_path = os.path.join(self.config.dumps.features, self.config.suffix, 'query.h5')
        self.archive_feat_path = os.path.join(self.config.dumps.features, self.config.suffix, 'archive.h5')
        self.retrieval_path = os.path.join(self.config.dumps.features, self.config.suffix, 'retrieval.h5')
        
    def prep_feature_extraction(self):
        self.model = model_selector(self.config, self.summary_writer)
        self.query_dataset = CLASS_MAPPING[self.config.dataset](
            self.config.train_tfrecord_paths, self.config.batch_size, get_shuffle_buffer_size(self.config.dataset, is_training=False)).dataset
        self.archive_dataset = CLASS_MAPPING[self.config.dataset](
            self.config.test_tfrecord_paths, self.config.batch_size, get_shuffle_buffer_size(self.config.dataset, is_training=False)).dataset
        features_path = os.path.join(self.config.dumps.features, self.config.suffix)
        if not os.path.isdir(features_path):
            os.makedirs(features_path)

    def finish_retrieval(self):
        self.summary_writer.close()
        print('Retrieval is finished')

    def feature_extraction(self):
        print('feature extraction is started')
        if (not os.path.isfile(self.query_feat_path)) or (not os.path.isfile(self.archive_feat_path)):
            self.prep_feature_extraction()
            self.restore_weigths()

        if not os.path.isfile(self.query_feat_path):
            with timer_calc() as elapsed_time_feat_ext:
                query_patch_names = []
                query_labels = []
                query_features = []
                with h5py.File(self.query_feat_path, 'w') as hf:
                    for query_batch_id, query_batch in enumerate(self.query_dataset):
                        with timer_calc() as elapsed_time:
                            self.model.prep_batch(query_batch, training=False)
                            query_feat_batch = self.model.get_features(self.model.batch)
                            for name, label, feature in zip(self.model.batch['patch_name'].numpy(), self.model.batch['label'], query_feat_batch.numpy()):
                                query_patch_names.append(name[0])
                                query_labels.append(label)
                                query_features.append(feature)
                            print('a batch of query features is extracted within {:0.2f} seconds'.format(elapsed_time()))
                    hf.create_dataset('feature', data=query_features)
                    hf.create_dataset('label', data=query_labels)
                    hf.create_dataset('patch_name', data=query_patch_names,dtype=h5py.string_dtype(encoding='utf-8'))
                print('feature extraction is finished for query set within {:0.2f} seconds'.format(elapsed_time_feat_ext()))
            
        if not os.path.isfile(self.archive_feat_path):
            with timer_calc() as elapsed_time_feat_ext:
                archive_patch_names = []
                archive_labels = []
                archive_features = []
                with h5py.File(self.archive_feat_path, 'w') as hf:
                    for archive_batch_id, archive_batch in enumerate(self.archive_dataset):
                        with timer_calc() as elapsed_time:
                            self.model.prep_batch(archive_batch, training=False)
                            archive_feat_batch = self.model.get_features(self.model.batch)
                            for name, label, feature in zip(self.model.batch['patch_name'].numpy(), self.model.batch['label'], archive_feat_batch.numpy()):
                                archive_patch_names.append(name[0])
                                archive_labels.append(label)
                                archive_features.append(feature)
                            print('a batch of archive features is extracted within {:0.2f} seconds'.format(elapsed_time()))
                    hf.create_dataset('feature', data=archive_features)
                    hf.create_dataset('label', data=archive_labels)
                    hf.create_dataset('patch_name', data=archive_patch_names,dtype=h5py.string_dtype(encoding='utf-8'))
                print('feature extraction is finished for archive set within {:0.2f} seconds'.format(elapsed_time_feat_ext()))  

    def retrieval(self):
        print('retrieval is started')

        if not os.path.isfile(self.retrieval_path):
            import numpy as np
            import psutil
            import ray
            from sklearn.metrics import pairwise_distances
            from tqdm import tqdm
            num_cpus = psutil.cpu_count()

            @ray.remote
            def calc_distance(query_feats, archive_feats):
                return pairwise_distances(query_feats, archive_feats, metric = lambda u, v: 0.5 * np.sum(((u - v) ** 2) / (u + v + 1e-10)))

            def batch_with_index(iterable, n=1):
                l = len(iterable)
                for ndx in range(0, l, n):
                    yield [iterable[ndx:min(ndx + n, l)], np.arange(ndx, min(ndx + n, l))]

            BATCH_SIZE = 1000
            with timer_calc() as elapsed_time:
                with h5py.File(self.archive_feat_path, 'r') as hf:
                    archive_feats = np.array(hf['feature'])

                with h5py.File(self.query_feat_path, 'r') as hf:
                    query_feats = np.array(hf['feature'])
                print('preparing data within {:0.2f} seconds'.format(elapsed_time()))

            with timer_calc() as elapsed_time:
                with h5py.File(self.retrieval_path, 'w') as hf:
                    distance_ds = hf.create_dataset('distance',(len(query_feats), len(archive_feats)), dtype='float32')
                    retrieval_res_ds = hf.create_dataset('retrieval_result', (len(query_feats), len(archive_feats)), dtype='int32')
                    pbar = tqdm(total=int(np.floor(len(query_feats)/float(BATCH_SIZE))))
                    for query_batch, query_batch_idx in batch_with_index(query_feats, BATCH_SIZE):
                        ray.init(num_cpus=num_cpus, object_store_memory = 30 * 1024 * 1024 * 1024)
                        result_ids = []
                        for archive_batch, archive_batch_idx in batch_with_index(archive_feats, BATCH_SIZE):
                            result_ids.append(calc_distance.remote(query_batch, archive_batch))
                        distance_batch = ray.get(result_ids)
                        distance_batch = np.concatenate(distance_batch, axis=1)
                        distance_ds[query_batch_idx] = distance_batch
                        retrieval_res_ds[query_batch_idx] = np.argsort(distance_batch, axis=-1)
                        pbar.update(1)
                        del result_ids
                        del distance_batch
                        ray.shutdown()
                    pbar.close()
                    print('calculating distance within {:0.2f} seconds'.format(elapsed_time()))

    def prep_metrics(self):
        print('metric preparation is started')
        import numpy as np

        with timer_calc() as elapsed_time:
            with h5py.File(self.archive_feat_path, 'r') as archive_feat_hf:
                with h5py.File(self.query_feat_path, 'r') as query_feat_hf:
                    archive_names = list(archive_feat_hf['patch_name'])
                    archive_labels = np.array(archive_feat_hf['label'])
                    query_names = list(query_feat_hf['patch_name'])
                    query_labels = np.array(query_feat_hf['label'])

            print('opening files within {:0.2f} seconds'.format(elapsed_time()))

        print('calc metrics')
        
        def nb_shared_labels_fnc(x, y):
            return len(set(np.where(x)[0]).intersection(np.where(y)[0]))

        @ray.remote
        def single_query_metric(max_topk, query_multi_hot, retrieved):
            retrieved_labels = []
            for i in range(len(retrieved)):
                if i < max_topk:
                    retrieved_labels.append(retrieved[i])
                else:
                    break

            retrieved_labels = np.array(retrieved_labels)
            normalized_discounted_cumulative_gains = np.zeros(max_topk)
            discounted_cumulative_gains = np.zeros(max_topk)
            max_discounted_cumulative_gains = np.zeros(max_topk)
            nb_shared_labels = np.array([nb_shared_labels_fnc(query_multi_hot, retrieved_labels[i]) for i in range(max_topk)])
            nb_shared_labels_ideal = -np.sort(-nb_shared_labels)
            for topk in range(1, max_topk+1):
                discounted_cumulative_gains[topk-1] = (2**nb_shared_labels[topk-1] - 1) / np.log2(1 + topk)
                max_discounted_cumulative_gains[topk-1] = (2**nb_shared_labels_ideal[topk-1] - 1) / np.log2(1 + topk)
                normalized_discounted_cumulative_gains[topk-1] = np.sum(discounted_cumulative_gains[:topk]) / np.sum(max_discounted_cumulative_gains[:topk])
            return normalized_discounted_cumulative_gains

        max_topk = 1000 if self.config.dataset == 'BEN' else 420
        
        normalized_discounted_cumulative_gains = np.zeros(max_topk)
        with h5py.File(self.retrieval_path, 'r+') as hf:
            for key in hf.keys():
                if not key == 'distance':
                    del hf[key]
            distance = hf['distance']
            import psutil
            num_cpus = psutil.cpu_count()
            ray.init(num_cpus=num_cpus, object_store_memory = 30 * 1024 * 1024 * 1024)
            result_ids = []
            for j in range(len(query_names)):
                query_multi_hot = query_labels[j]
                ins_distance = distance[j]
                ins_sorted_distance = np.argsort(ins_distance)
                retrieved = ins_sorted_distance[range(max_topk)]
                result_ids.append(single_query_metric.remote(max_topk, query_multi_hot, archive_labels[retrieved]))

            with timer_calc() as elapsed_time:
                with tqdm(total=len(query_names)) as pbar:
                    while True:
                        ready, not_ready = ray.wait(result_ids)
                        nb_done = len(ready)
                        scores = np.array(ray.get(ready))
                        pbar.update(nb_done)
                        normalized_discounted_cumulative_gains += np.sum(scores, axis=0)
                        result_ids = not_ready
                        if not result_ids:
                            break
                    ray.shutdown()
                print('{} tasks finished within {:0.2f} seconds'.format(len(query_names), elapsed_time()))

            normalized_discounted_cumulative_gains /= float(len(query_names))
            NDCG = normalized_discounted_cumulative_gains[29 if self.config.dataset == 'BEN' else 19]
            hf.create_dataset('NDCG', data=NDCG)
        
    def restore_weigths(self):
        self.model.neural_net = tf.keras.models.load_model(os.path.join(self.config.dumps.model_weights, self.config.suffix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= 'Training script')
    parser.add_argument('configs', help= 'json config file', nargs = '+')
    parser_args = parser.parse_args()

    for args_parsed in parser_args.configs:
        with timer_calc() as elapsed_time:
            with open(args_parsed, 'r') as fp:
                config_dict = json.load(fp)
            config = Config(config_dict, training=False)
            if (config.dataset == 'BEN') and (config.model_name == 'GRID-PCE' or config.model_name == 'GRID-RRL'):
                print('BEN can be used with GRID-BCE')
            else:
                select_gpu(config.gpu)
                retrieval = Retrieval(config)

                retrieval.feature_extraction()
                retrieval.retrieval()
                retrieval.prep_metrics()
                retrieval.finish_retrieval()
                del retrieval
                del config
                print('{} is finished within {:0.2f} seconds'.format(args_parsed, elapsed_time()))