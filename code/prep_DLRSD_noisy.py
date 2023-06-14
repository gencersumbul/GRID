import os
import tensorflow as tf
import cv2
from pandas import read_excel
import numpy as np
np.random.seed(42)
from tqdm import tqdm
import argparse

test_ratio = 0.2
val_ratio = 0.1
crop_size = 224
scale_size = 256
n_classes = 17

DLRSD_lookup = {
    'airplane': {'id':0, 'val': [ 166, 202, 240 ]},
    'baresoil': {'id':1, 'val':  [ 128, 128, 0 ]},
    'buildings': {'id':2, 'val':  [ 0, 0, 128 ]},
    'cars': {'id':3, 'val':  [ 255, 0, 0 ]},
    'chaparral': {'id':4, 'val':  [ 0, 128, 0 ]},
    'court': {'id':5, 'val':  [ 128, 0, 0 ]},
    'dock': {'id':6, 'val':  [ 255, 233, 233 ]},
    'field': {'id':7, 'val':  [ 160, 160, 164 ]},
    'grass': {'id':8, 'val':  [ 0, 128, 128]},
    'mobilehome': {'id':9, 'val':  [ 90, 87, 255]},
    'pavement': {'id':10, 'val':  [ 255, 255, 0 ]},
    'sand': {'id':11, 'val':  [ 255, 192, 0 ]},
    'sea': {'id':12, 'val':  [ 0, 0, 255, ]},
    'ship': {'id':13, 'val':  [ 255, 0 , 192 ]},
    'tanks': {'id':14, 'val':  [ 128, 0, 128 ]},
    'trees': {'id':15, 'val':  [ 0, 255, 0 ]},
    'water': {'id':16, 'val':  [ 0, 255, 255 ]}
}

def train_selector(s_range):
    indices = []
    for i in range(len(s_range)):
        if ((i + 1) % 10 == 2) or ((i + 1) % 10 == 7):
            pass
        elif (i + 1) % 10 == 5:
            pass
        else:
            indices.append(i)
    return indices

def change_class(orig_segmap, noisy_seg_map, label):
    new_seg_map = noisy_seg_map.copy()
    label_idx = np.where(np.arange(17) == label)[0][0]
    labels = np.delete(np.arange(17), label_idx)
    new_label = np.random.choice(labels)
    new_seg_map[np.where(orig_segmap == label)] = new_label
    return new_seg_map

def convert_multi_hot(indices):
    multi_hot = np.zeros(17)
    multi_hot[indices] = 1
    return list(multi_hot)

def get_records(DLRSDPath, UCMercedPath, outPath, ucmerced_ext=".tif", DLRSD_ext=".png"):
    writer_train = tf.io.TFRecordWriter(os.path.join(outPath + '/TFRecord/', "train.tfrecord"))
    writer_test = tf.io.TFRecordWriter(os.path.join(outPath + '/TFRecord/', "test.tfrecord"))
    writer_val = tf.io.TFRecordWriter(os.path.join(outPath + '/TFRecord/', "val.tfrecord"))

    ucmerced_class_names = [f for f in os.listdir(os.path.join(UCMercedPath, 'Images')) if not f.startswith('.')]
    dlrsd_class_names = [f for f in os.listdir(os.path.join(DLRSDPath, 'Images')) if not f.startswith('.')]
    multi_label_dict = read_excel(os.path.join(DLRSDPath, 'multi-labels.xlsx'), sheet_name = 'Sheet1').set_index('IMAGE\LABEL').T.to_dict('list')

    for ucmerced_name, dlrsd_name in tqdm(zip(ucmerced_class_names, dlrsd_class_names)):
        ucmerced_directory = os.path.join(UCMercedPath, 'Images', ucmerced_name)
        dlrsd_directory = os.path.join(DLRSDPath, 'Images', dlrsd_name)
        ucmerced_class_image_paths = [os.path.join(ucmerced_directory, f) for f in os.listdir(ucmerced_directory) if f.endswith(ucmerced_ext)]
        DLRSD_class_image_paths = [os.path.join(dlrsd_directory, f) for f in os.listdir(dlrsd_directory) if f.endswith(DLRSD_ext)]

        for i, (ucmerced_img_path, DLRSD_img_path) in enumerate(zip(ucmerced_class_image_paths, DLRSD_class_image_paths)):
            patch_name = DLRSD_img_path.split(DLRSD_ext)[0].split('/')[-1]
            label = multi_label_dict[patch_name]
            ucmerced_img = cv2.imread(ucmerced_img_path)
            ucmerced_img = cv2.cvtColor(ucmerced_img, cv2.COLOR_BGR2RGB)
            ucmerced_img = cv2.resize(ucmerced_img, (256, 256), interpolation=cv2.INTER_CUBIC)

            DLRSD_img = cv2.imread(DLRSD_img_path)
            DLRSD_img = cv2.cvtColor(DLRSD_img, cv2.COLOR_BGR2RGB)
            seg_map = np.zeros((256, 256)) - 1
            for key in DLRSD_lookup.keys():
                seg_map[np.where((DLRSD_img == DLRSD_lookup[key]['val']).all(axis=2))] = DLRSD_lookup[key]['id']

            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=label)),
                "noisy_label": tf.train.Feature(float_list=tf.train.FloatList(value=label)),
                "img": tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(ucmerced_img))),
                "seg_map": tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(seg_map))),
                "noisy_seg_map": tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(seg_map))),
                "patch_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_name.encode('utf-8')]))
            }))
            if ((i + 1) % 10 == 2) or ((i + 1) % 10 == 7):
                writer_test.write(example.SerializeToString())
            elif (i + 1) % 10 == 5:
                writer_val.write(example.SerializeToString())
            else:
                writer_train.write(example.SerializeToString())

    writer_train.close()
    writer_test.close()
    writer_val.close()

def get_noisy_records(DLRSDPath, UCMercedPath, outPath, noise_prcnt, ucmerced_ext=".tif", DLRSD_ext=".png"):
    writer_train = tf.io.TFRecordWriter(os.path.join(outPath + '/TFRecord/', "train_noise_prcnt{}.tfrecord".format(noise_prcnt)))
    ucmerced_class_names = [f for f in os.listdir(os.path.join(UCMercedPath, 'Images')) if not f.startswith('.')]
    dlrsd_class_names = [f for f in os.listdir(os.path.join(DLRSDPath, 'Images')) if not f.startswith('.')]
    multi_label_dict = read_excel(os.path.join(DLRSDPath, 'multi-labels.xlsx'), sheet_name = 'Sheet1').set_index('IMAGE\LABEL').T.to_dict('list')
    for ucmerced_name, dlrsd_name in tqdm(zip(ucmerced_class_names, dlrsd_class_names)):
        ucmerced_directory = os.path.join(UCMercedPath, 'Images', ucmerced_name)
        dlrsd_directory = os.path.join(DLRSDPath, 'Images', dlrsd_name)
        ucmerced_class_image_paths = [os.path.join(ucmerced_directory, f) for f in os.listdir(ucmerced_directory) if f.endswith(ucmerced_ext)]
        DLRSD_class_image_paths = [os.path.join(dlrsd_directory, f) for f in os.listdir(dlrsd_directory) if f.endswith(DLRSD_ext)]
        sample_idx = np.arange(len(ucmerced_class_image_paths))
        train_indices = train_selector(sample_idx)       
        label_arr = np.zeros((len(ucmerced_class_image_paths), 17))
        for i, (ucmerced_img_path, DLRSD_img_path) in enumerate(zip(ucmerced_class_image_paths, DLRSD_class_image_paths)):
            if i in train_indices:
                patch_name = DLRSD_img_path.split(DLRSD_ext)[0].split('/')[-1]
                label_arr[i] = multi_label_dict[patch_name]
        label_indices = np.where(label_arr)
        label_indices = np.array([[i,j] for i,j in zip(label_indices[0],label_indices[1])])
        noisy_indices = label_indices[np.random.choice(range(label_indices.shape[0]), int(len(label_indices)*noise_prcnt*0.01), replace=False)]
        noisy_row_indices = noisy_indices[:,0]
        noisy_col_indices = noisy_indices[:,1]
        for i, (ucmerced_img_path, DLRSD_img_path) in enumerate(zip(ucmerced_class_image_paths, DLRSD_class_image_paths)):
            if i in train_indices:
                patch_name = DLRSD_img_path.split(DLRSD_ext)[0].split('/')[-1]
                label = multi_label_dict[patch_name]
                ucmerced_img = cv2.imread(ucmerced_img_path)
                ucmerced_img = cv2.cvtColor(ucmerced_img, cv2.COLOR_BGR2RGB)
                ucmerced_img = cv2.resize(ucmerced_img, (256, 256), interpolation=cv2.INTER_CUBIC)

                DLRSD_img = cv2.imread(DLRSD_img_path)
                DLRSD_img = cv2.cvtColor(DLRSD_img, cv2.COLOR_BGR2RGB)
                seg_map = np.zeros((256, 256)) - 1
                for key in DLRSD_lookup.keys():
                    seg_map[np.where((DLRSD_img == DLRSD_lookup[key]['val']).all(axis=2))] = DLRSD_lookup[key]['id']

                noisy_seg_map = seg_map.copy()
                if i in noisy_row_indices:
                    labels = np.unique(seg_map)
                    if -1 in labels:
                        non_idx = np.where(labels == -1)[0][0]
                        labels = np.delete(labels, non_idx)
                    labels_to_be_changed = noisy_col_indices[np.where(noisy_row_indices == i)[0]]
                    for img_label in labels:
                        if img_label in labels_to_be_changed:
                            noisy_seg_map = change_class(seg_map, noisy_seg_map, img_label)

                noisy_label = np.unique(noisy_seg_map).astype(np.int)
                if -1 in noisy_label:
                    non_idx = np.where(noisy_label == -1)[0][0]
                    noisy_label = np.delete(noisy_label, non_idx)
                noisy_label = convert_multi_hot(noisy_label)

                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=label)),
                    "noisy_label": tf.train.Feature(float_list=tf.train.FloatList(value=noisy_label)),
                    "img": tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(ucmerced_img))),
                    "seg_map": tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(seg_map))),
                    "noisy_seg_map": tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(noisy_seg_map))),
                    "patch_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_name.encode('utf-8')]))
                }))
                writer_train.write(example.SerializeToString())
    writer_train.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'This script creates TFRecord files for DLRSD archive')
    parser.add_argument('-r_dlrsd', '--root_DLRSD_folder', dest = 'root_DLRSD_folder', default = '../dumps/data/DLRSD', help = 'root folder path contains multiple patch folders')
    parser.add_argument('-r_ucm', '--root_UCM_folder', dest = 'root_UCM_folder', default = '../dumps/data/UCMerced_LandUse', help = 'root folder path contains multiple patch folders')
    parser.add_argument('-o', '--out_folder', dest='out_folder', default = '../dumps/data/DLRSD', help='folder path containing result files')

    args = parser.parse_args()

    get_records(args.root_DLRSD_folder, args.root_UCM_folder, args.out_folder)
    for noise_prcnt in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        get_noisy_records(args.root_DLRSD_folder, args.root_UCM_folder, args.out_folder, noise_prcnt=noise_prcnt)
