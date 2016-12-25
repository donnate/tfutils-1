"""
The is the basic illustration of training.
"""
from __future__ import division, print_function, absolute_import
import os, sys, math, time
from datetime import datetime
import pymongo as pm
import numpy as np

import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from tfutils import base, model, utils

class MNIST(object):
    def __init__(self, batch_size=100, group='train'):
        """
        A specific reader for MNIST stored as a HDF5 file

        Args:
            - data_path: path to imagenet data
            - crop_size: for center crop (crop_size x crop_size)
            - *args: extra arguments for HDF5DataProvider
        Kwargs:
            - **kwargs: extra keyword arguments for HDF5DataProvider
        """
        self.batch_size = batch_size
        data = read_data_sets('/tmp')
        if group == 'train':
            self.data = data.train
        elif group == 'test':
            self.data = data.train
        elif group == 'validation':
            self.data = data.train
        else:
            raise ValueError('MNIST data input "{}" not known'.format(group))

        self.node = {'images': tf.placeholder(tf.float32,
                                              shape=(self.batch_size, 784),
                                              name='images'),
                     'labels': tf.placeholder(tf.int32,
                                              shape=[self.batch_size],
                                              name='labels')}

    def __iter__(self):
        return self

    def next(self):
        batch = self.data.next_batch(self.batch_size)
        feed_dict = {self.node['images']: batch[0],
                     self.node['labels']: batch[1]}
        return feed_dict


num_batches_per_epoch = 10000//256
testhost = 'localhost'
testport = 31001
testdbname = 'tfutils-test'
testcol = 'testcol'

def test_training():
    params = {}
    params['model_params'] = {'func': model.mnist_tfutils}
    params['save_params'] = {'host': testhost,
                             'port': testport,
                             'dbname': testdbname,
                             'collname': testcol,
                             'exp_id': 'training0',
                             'save_valid_freq': 20,
                             'save_filters_freq': 200,
                             'cache_filters_freq': 100}
    params['train_params'] = {'data': {'func': MNIST,
                                       'batch_size': 100,
                                       'group': 'train'},
                              'queue_params': {'queue_type': 'fifo',
                                               'batch_size': 100,
                                               'n_threads': 4}}
    params['learning_rate_params'] = {'learning_rate': 0.05,
                                      'decay_steps': num_batches_per_epoch,
                                      'decay_rate': 0.95,
                                      'staircase': True}
    params['num_steps'] = 500

    conn = pm.MongoClient(host=testhost,
                          port=testport)
    conn.drop_database(testdbname)
    nm = testdbname + '_' + testcol + '_training0'
    [conn.drop_database(x) for x in conn.database_names() if x.startswith(nm) and '___RECENT' in x]
    nm = testdbname + '_' + testcol + '_training1'
    [conn.drop_database(x) for x in conn.database_names() if x.startswith(nm) and '___RECENT' in x]

    base.train_from_params(**params)

    assert conn[testdbname][testcol+'.files'].find({'exp_id': 'training0'}).count() == 26
    assert conn[testdbname][testcol+'.files'].find({'exp_id': 'training0', 
            'saved_filters': True}).distinct('step') == [0, 200, 400]

    params['num_steps'] = 1000
    base.train_from_params(**params)

    assert conn[testdbname][testcol+'.files'].find({'exp_id': 'training0'}).count() == 51
    assert conn[testdbname][testcol+'.files'].find({'exp_id': 'training0', 
         'saved_filters': True}).distinct('step') == [0, 200, 400, 600, 800, 1000]

    assert conn['tfutils-test']['testcol.files'].distinct('exp_id') == ['training0']

    params['num_steps'] = 1500
    params['load_params'] = {'exp_id': 'training0'}
    params['save_params']['exp_id'] = 'training1'
    base.train_from_params(**params)
    
    assert conn[testdbname][testcol+'.files'].find({'exp_id': 'training1', 
                            'saved_filters': True}).distinct('step') == [1200, 1400]


def test_validation():
    params = {}
    params['model_params'] = {'func': model.mnist_tfutils}
    params['load_params'] = {'host': testhost,
                             'port': testport,
                             'dbname': testdbname,
                             'collname': testcol,
                             'exp_id': 'training0'}
    params['save_params'] = {'exp_id': 'validation0'}
    params['validation_params'] = {'valid0': {'data': {'func': MNIST,
                                                       'batch_size': 100,
                                                       'group': 'train'
                                                   },
                                              'queue_params': {'queue_type': 'fifo',
                                                               'batch_size': 100,
                                                               'n_threads': 4},
                                              'num_steps': 10,
                                              'agg_func': utils.mean_dict}}

    base.test_from_params(**params)

    conn = pm.MongoClient(host=testhost,
                          port=testport)
    assert conn[testdbname][testcol+'.files'].find({'exp_id': 'validation0'}).count() == 1
    r = conn[testdbname][testcol+'.files'].find({'exp_id': 'validation0'})[0]
    assert r['validation_only'] == True
    f = r['validation_results']['valid0']['loss']
    idval = conn[testdbname][testcol+'.files'].find({'exp_id': 'training0'})[50]['_id']
    v = conn[testdbname][testcol+'.files'].find({'exp_id': 'validation0'})[0]['validates']
    assert idval == v


def get_extraction_target(inputs, outputs, **params):
    """here's how to figure out what names to use:
    names = [[x.name for x in op.values()] for op in tf.get_default_graph().get_operations()]
    print("NAMES", names)
    """
    f = tf.get_default_graph().get_tensor_by_name('validation/valid1/hidden1/fc:0')
    targets = {'loss': utils.get_loss(inputs, outputs, **params),
               'features': f}
    return targets


def test_feature_extraction():
    params = {}
    params['model_params'] = {'func': model.mnist_tfutils}
    params['load_params'] = {'host': testhost,
                             'port': testport,
                             'dbname': testdbname,
                             'collname': testcol,
                             'exp_id': 'training0'}
    params['save_params'] = {'exp_id': 'validation1',
                             'save_intermediate_freq': 1,
                             'save_to_gfs': ['features']}

    targdict = {'func': get_extraction_target}
    targdict.update(base.default_loss_params())
    params['validation_params'] = {'valid1': {'data': {'func': MNIST,
                                                     'batch_size': 100,
                                                     'group': 'train'
                                                 },
                                            'targets': targdict,
                                            'queue_params': {'queue_type': 'fifo',
                                                             'batch_size': 100,
                                                             'n_threads': 4},
                                            'num_steps': 10,
                                            'online_agg_func': utils.reduce_mean_dict
                                            }
                                   }
    base.test_from_params(**params)

    conn = pm.MongoClient(host=testhost,
                          port=testport)
    coll = conn[testdbname][testcol+'.files']
    assert coll.find({'exp_id': 'validation1'}).count() == 11
    q = {'exp_id': 'validation1', 'validation_results.valid1.intermediate_steps': {'$exists': True}}
    assert coll.find(q).count() == 1
    r = coll.find(q)[0]
    q1 = {'exp_id': 'validation1', 'validation_results.valid1.intermediate_steps': {'$exists': False}}
    ids = coll.find(q1).distinct('_id')
    assert r['validation_results']['valid1']['intermediate_steps'] == ids

    


