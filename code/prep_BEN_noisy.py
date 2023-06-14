import argparse
import os
import csv
import json
from tqdm import tqdm
import tensorflow as tf
import numpy as np

GDAL_EXISTED = False
RASTERIO_EXISTED = False

def prep_example(bands, BEN_19_labels, BEN_19_multi_hot, BEN_19_multi_hot_noisy, patch_name):
    return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'B01': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(bands['B01']))),
                    'B02': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(bands['B02']))),
                    'B03': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(bands['B03']))),
                    'B04': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(bands['B04']))),
                    'B05': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(bands['B05']))),
                    'B06': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(bands['B06']))),
                    'B07': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(bands['B07']))),
                    'B08': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(bands['B08']))),
                    'B8A': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(bands['B8A']))),
                    'B09': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(bands['B09']))),
                    'B11': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(bands['B11']))),
                    'B12': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(bands['B12']))),
                    'BEN-19_labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[i.encode('utf-8') for i in BEN_19_labels])),
                    'BEN-19_multi_hot': tf.train.Feature(float_list=tf.train.FloatList(value=BEN_19_multi_hot)),
                    'noisy_BEN-19_multi_hot': tf.train.Feature(float_list=tf.train.FloatList(value=BEN_19_multi_hot_noisy)),
                    'patch_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_name.encode('utf-8')]))
                }))

def read_label(root_folder, patch_name, label_indices):
    patch_folder_path = os.path.join(root_folder, patch_name)
    original_multi_hot = np.zeros(
        len(label_indices['original_labels'].keys()), dtype=int)
    BEN_19_multi_hot = np.zeros(len(label_indices['label_conversion']),dtype=int)
    patch_json_path = os.path.join(
        patch_folder_path, patch_name + '_labels_metadata.json')

    with open(patch_json_path, 'rb') as f:
        patch_json = json.load(f)

    original_labels = patch_json['labels']
    for label in original_labels:
        original_multi_hot[label_indices['original_labels'][label]] = 1

    for i in range(len(label_indices['label_conversion'])):
        BEN_19_multi_hot[i] = (
                np.sum(original_multi_hot[label_indices['label_conversion'][i]]) > 0
            ).astype(int)
    return BEN_19_multi_hot

def prep_tf_record(root_folder, split_name, csv_file, noise_prcnt, out_folder, label_indices, BEN_19_label_idx, GDAL_EXISTED, RASTERIO_EXISTED):
    if GDAL_EXISTED:
        import gdal
    elif RASTERIO_EXISTED:
        import rasterio

    band_names = ['B01', 'B02', 'B03', 'B04', 'B05',
                'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    noise_suffix = '' if noise_prcnt == 0 else '_noise_prcnt{}'.format(noise_prcnt)
    TFRecord_writer = tf.io.TFRecordWriter(out_folder + '/TFRecord/' + split_name + noise_suffix + '.tfrecord')

    patch_names = []
    with open(csv_file) as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        for j in csv_reader:
            if len(j) > 0:
                patch_names.append(j[0])
    
    label_matrix = []

    if noise_prcnt == 0:
        noisy_row_indices = []; noisy_col_indices = []
    else:
        for patch_name in tqdm(patch_names, desc = "read labels for {} split with {}% noise".format(split_name, noise_prcnt), total=len(patch_names)):
            BEN_19_multi_hot = read_label(root_folder, patch_name, label_indices)
            label_matrix.append(BEN_19_multi_hot)

        label_matrix = np.array(label_matrix)
        label_indices = np.where(label_matrix)
        label_indices = np.array([[i,j] for i,j in zip(label_indices[0],label_indices[1])])
        noisy_indices = label_indices[np.random.choice(range(label_indices.shape[0]), int(len(label_indices)*noise_prcnt*0.01), replace=False)]
        noisy_row_indices = np.array(noisy_indices[:,0])
        noisy_row_indices = label_matrix[noisy_row_indices]
        noisy_col_indices = noisy_indices[:,1]

    for patch_idx, patch_name in tqdm(enumerate(patch_names), desc = "prep records for {} split with {}% noise".format(split_name, noise_prcnt), total=len(patch_names)):
        patch_folder_path = os.path.join(root_folder, patch_name)
        bands = {}
        for band_name in band_names:
            # First finds related GeoTIFF path and reads values as an array
            band_path = os.path.join(
                patch_folder_path, patch_name + '_' + band_name + '.tif')
            if GDAL_EXISTED:
                band_ds = gdal.Open(band_path,  gdal.GA_ReadOnly)
                raster_band = band_ds.GetRasterBand(1)
                band_data = raster_band.ReadAsArray()
                bands[band_name] = np.array(band_data)
            elif RASTERIO_EXISTED:
                band_ds = rasterio.open(band_path)
                band_data = np.array(band_ds.read(1))
                bands[band_name] = np.array(band_data)
        
        if len(label_matrix) == 0:
            BEN_19_multi_hot = read_label(root_folder, patch_name, label_indices)
        else:
            BEN_19_multi_hot = label_matrix[patch_idx]

        BEN_19_multi_hot_noisy = BEN_19_multi_hot.copy()
        BEN_19_labels = []
        for i in np.where(BEN_19_multi_hot == 1)[0]:
            BEN_19_labels.append(BEN_19_label_idx[i])
        
        if i in noisy_row_indices:
            labels_to_be_changed = noisy_col_indices[np.where(noisy_row_indices == i)[0]]
            for label_idx, label in enumerate(BEN_19_multi_hot):
                if label_idx in labels_to_be_changed:
                    BEN_19_multi_hot_noisy[label_idx] = 0
                    BEN_19_multi_hot_noisy[np.random.choice(np.delete(np.arange(19),label_idx))] = 1

        example = prep_example(
            bands, 
            BEN_19_labels,
            BEN_19_multi_hot,
            BEN_19_multi_hot_noisy,
            patch_name
        )
        TFRecord_writer.write(example.SerializeToString())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= 'This script creates TFRecord files for BigEarthNet archive')
    parser.add_argument('-r', '--root_folder', dest = 'root_folder', help = 'root folder path contains multiple patch folders')
    parser.add_argument('-o', '--out_folder', dest='out_folder', default = '../dumps/data/BEN', help='folder path containing result files')
    parser.add_argument('-train', '--train_csv', dest='train_csv', default = '../dumps/data/BEN/train.csv', help='csv file containing patch names')
    parser.add_argument('-val', '--val_csv', dest='val_csv', default = '../dumps/data/BEN/val.csv', help='csv file containing patch names')
    parser.add_argument('-test', '--test_csv', dest='test_csv', default = '../dumps/data/BEN/train.csv', help='csv file containing patch names')

    args = parser.parse_args()

    try:
        import gdal
        GDAL_EXISTED = True
        print('INFO: GDAL package will be used to read GeoTIFF files')
    except ImportError:
        try:
            import rasterio
            RASTERIO_EXISTED = True
            print('INFO: rasterio package will be used to read GeoTIFF files')
        except ImportError:
            print('ERROR: please install either GDAL or rasterio package to read GeoTIFF files')
            exit()

    with open('../dumps/data/BEN/label_indices.json', 'rb') as f:
        label_indices = json.load(f)

    BEN_19_label_idx = {v: k for k, v in label_indices['BEN-19_labels'].items()}

    for noise_prcnt in range(0,70,10):
        prep_tf_record(
                os.path.realpath(args.root_folder),
                'train', 
                os.path.realpath(args.train_csv), 
                noise_prcnt,
                os.path.realpath(args.out_folder),
                label_indices,
                BEN_19_label_idx,
                GDAL_EXISTED, 
                RASTERIO_EXISTED
            )

    prep_tf_record(
            os.path.realpath(args.root_folder),
            'val', 
            os.path.realpath(args.val_csv), 
            0,
            os.path.realpath(args.out_folder),
            label_indices,
            BEN_19_label_idx,
            GDAL_EXISTED, 
            RASTERIO_EXISTED
        )

    prep_tf_record(
            os.path.realpath(args.root_folder),
            'test', 
            os.path.realpath(args.test_csv), 
            0,
            os.path.realpath(args.out_folder),
            label_indices,
            BEN_19_label_idx,
            GDAL_EXISTED, 
            RASTERIO_EXISTED
        )