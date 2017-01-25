"""
Script used for training the normalnet using tfutils
"""

from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf

from tfutils import base, data, model, optimizer

import json
import copy

#os.environ["CUDA_VISIBLE_DEVICES"]="2"

host = os.uname()[1]
if host == 'kanefsky':  # kanefsky
    DATA_PATH = '/mnt/data/imagenet_omind7/data.raw'
elif host == 'freud':  # freud
    DATA_PATH = '/media/data2/one_world_dataset/data.raw'
else:
    print("Not supported yet!")
    exit()

def in_top_k(inputs, outputs, target):
    return {'top1': tf.nn.in_top_k(outputs, inputs[target], 1),
            'top5': tf.nn.in_top_k(outputs, inputs[target], 5)}

def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k:[] for k in res}
    for k,v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res


def exponential_decay(global_step,
                      learning_rate=.01,
                      decay_factor=.95,
                      decay_steps=1,
                      ):
    # Decay the learning rate exponentially based on the number of steps.
    if decay_factor is None:
        lr = learning_rate  # just a constant.
    else:
        # Calculate the learning rate schedule.
        lr = tf.train.exponential_decay(
            learning_rate,  # Base learning rate.
            global_step,  # Current index into the dataset.
            decay_steps,  # Decay step
            decay_factor,  # Decay rate.
            staircase=True)
    return lr


class ImageNetSame(data.HDF5DataProvider):

    N_TRAIN = 1290129 - 50000
    N_VAL = 50000

    def __init__(self,
                 data_path,
                 group='train',
                 batch_size=1,
                 crop_size=None,
                 *args,
                 **kwargs):

        self.group = group
        self.images = 'data'
        self.labels = 'labels'
        if self.group=='train':
            subslice = range(self.N_TRAIN)
        else:
            subslice = range(self.N_TRAIN, self.N_TRAIN + self.N_VAL)
        super(ImageNetSame, self).__init__(
            data_path,
            [self.images, self.labels],
            batch_size=batch_size,
            postprocess={self.images: self.postproc_imgs, self.labels: self.postproc_labels},
            pad=True,
            subslice=subslice,
            *args, **kwargs)
        if crop_size is None:
            self.crop_size = 256
        else:
            self.crop_size = crop_size

        self.reshape_size = [batch_size, 3, 256, 256]

    def postproc_imgs(self, ims, f):
        norm = ims.astype(np.float32) / 255
        norm = norm.reshape(self.reshape_size)
        norm = np.transpose(norm, (0, 2, 3, 1))
        if self.group=='train':
            off = np.random.randint(0, 256 - self.crop_size, size=2)
        else:
            off = int((256 - self.crop_size)/2)
            off = [off, off]
        images_batch = norm[:,
                            off[0]: off[0] + self.crop_size,
                            off[1]: off[1] + self.crop_size]

        return images_batch

    def postproc_labels(self, labels, f):
        return labels.astype(np.int32)

    def next(self):
        batch = super(ImageNetSame, self).next()
        feed_dict = {'images': np.squeeze(batch[self.images]),
                     'labels': np.squeeze(batch[self.labels])}
        return feed_dict

def loss_with_postitional_para(outputs, targets, *args, **loss_func_kwargs):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targets, logits = outputs)

BATCH_SIZE = 256
NUM_BATCHES_PER_EPOCH = ImageNetSame.N_TRAIN // BATCH_SIZE
IMAGE_SIZE_CROP = 224
NUM_CHANNELS = 3

def alexnet(inputs, train=True, norm=True, **kwargs):
    m = model.ConvNet(**kwargs)
    dropout = .5 if train else None

    with tf.contrib.framework.arg_scope([m.conv], init='xavier',
                                        stddev=.01, bias=0, activation='relu'):
        with tf.variable_scope('conv1'):
            m.conv(96, 11, 4, padding='VALID', in_layer=inputs)
            if norm:
                m.norm(depth_radius=5, bias=1, alpha=.0001, beta=.75)
            m.pool(3, 2)

        with tf.variable_scope('conv2'):
            m.conv(256, 5, 1)
            if norm:
                m.norm(depth_radius=5, bias=1, alpha=.0001, beta=.75)
            m.pool(3, 2)

        with tf.variable_scope('conv3'):
            m.conv(384, 3, 1)

        with tf.variable_scope('conv4'):
            m.conv(256, 3, 1)

        with tf.variable_scope('conv5'):
            m.conv(256, 3, 1)
            m.pool(3, 2)

        with tf.variable_scope('fc6'):
            m.fc(4096, init='trunc_norm', dropout=dropout, bias=.1)

        with tf.variable_scope('fc7'):
            m.fc(4096, init='trunc_norm', dropout=dropout, bias=.1)

        with tf.variable_scope('fc8'):
            m.fc(1000, init='trunc_norm', activation=None, dropout=None, bias=0)

    return m

def alexnet_tfutils(inputs, **kwargs):
    m = alexnet(inputs['images'], **kwargs)
    return m.output, m.params

def main():
    params = {
        'save_params': {
            'host': 'localhost',
            #'port': 31001,
            #'port': 22334,
            'port': 27017,
            'dbname': 'alexnet-kanefbenchtest',
            'collname': 'alexnet',
            #'exp_id': 'trainval0',
            'exp_id': 'trainval0',
            #'exp_id': 'trainval2', # using screen?

            'do_save': True,
            #'do_save': False,
            'save_initial_filters': True,
            'save_metrics_freq': 2000,  # keeps loss from every SAVE_LOSS_FREQ steps.
            'save_valid_freq': 5000,
            'save_filters_freq': 20000,
            'cache_filters_freq': 5000,
            # 'cache_dir': None,  # defaults to '~/.tfutils'
        },

        'load_params': {
            'host': 'localhost',
            # 'port': 31001,
            # 'dbname': 'alexnet-test',
            # 'collname': 'alexnet',
            # 'exp_id': 'trainval0',
            #'port': 22334,
            'port': 27017,
            'dbname': 'alexnet-kanefbenchtest',
            'collname': 'alexnet',
            #'exp_id': 'trainval0',
            'exp_id': 'trainval0',
            #'exp_id': 'trainval2', # using screen?
            'do_restore': False,
            'load_query': None
        },

        'model_params': {
            'func': alexnet_tfutils,
            'seed': 0,
            'norm': False  # do you want local response normalization?
        },

        'train_params': {
            'data_params': {
                'func': ImageNetSame,
                'data_path': DATA_PATH,
                'group': 'train',
                'crop_size': IMAGE_SIZE_CROP,
                'batch_size': 1
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': BATCH_SIZE,
                'n_threads': 4,
                'seed': 0,
            },
            'thres_loss': 1000,
            'num_steps': 90 * NUM_BATCHES_PER_EPOCH  # number of steps to train
        },

        'loss_params': {
            'targets': 'labels',
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
            #'loss_per_case_func': loss_with_postitional_para,
        },

        'learning_rate_params': {
            'func': tf.train.exponential_decay,
            'learning_rate': .01,
            'decay_rate': .95,
            'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
            'staircase': True
        },

        'optimizer_params': {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.MomentumOptimizer,
            'clip': True,
            'momentum': .9
        },
        'validation_params': {
            'topn': {
                'data_params': {
                    'func': ImageNetSame,
                    'data_path': DATA_PATH,  # path to image database
                    'group': 'val',
                    'crop_size': IMAGE_SIZE_CROP,  # size after cropping an image
                },
                'targets': {
                    'func': in_top_k,
                    'target': 'labels',
                },
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': BATCH_SIZE,
                    'n_threads': 4,
                    'seed': 0,
                },
                'num_steps': ImageNetSame.N_VAL // BATCH_SIZE + 1,
                'agg_func': lambda x: {k:np.mean(v) for k,v in x.items()},
                'online_agg_func': online_agg
            },
        },

        'log_device_placement': False,  # if variable placement has to be logged
    }
    base.get_params()
    base.train_from_params(**params)

if __name__ == '__main__':
    #base.get_params()
    #base.train_from_params(**params)
    main()
