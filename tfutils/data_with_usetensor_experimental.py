from __future__ import absolute_import, division, print_function

import threading
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

class TFRecordsDataProvider(object):
    def __init__(self,
		 tfsource,
		 sourcelist,
		 batch_size,
		 preprocess=None,
		 postprocess=None,
		 pad=False,
		 decodelist=None,
		 usetensor=False,
		):
        """
        - tfrecsource (str): path where tfrecords file resides
        - sourcelist (list of strs): list of keys in the tfrecords file to use as source dataarrays
        - batch_size (int): size of batches to be returned
        - preprocess (dict of callables): functions for preprocessing data in the datasources. Keys of this are subset of sourcelist. Attention: preprocessing on tensors that have to be read in first by yourself!
        - postprocess (dict of callables): functions for postprocess data.  Keys of this are subset of sourcelist.
        - pad (bool): whether to pad data returned if amount of data left to return is less then full batch size
	- decodelist (list of strs): list of keys in the tfrecords file that have to be decoded from raw bytes format and reshaped to their original form, e. g. numpy arrays or serialized images
	- usetensor (bool): whether to output the data as a tensorflow tensor or numpy array
        """
        self.tfsource = tfsource
        self.file = tf.python_io.tf_record_iterator(path=self.tfsource)
        self.sourcelist = sourcelist
        self.subslice = subslice
        self.subsliceinds = None
        self.preprocess = {} if preprocess is None else preprocess
        self.postprocess = {} if postprocess is None else postprocess
	self.pad = pad
        self.decodelist = [] if decodelist is None else decodelist
	self.usetensor = usetensor

        for source in self.decodelist:
            assert source in self.sourcelist, 'decodelist has to be a subset of sourcelist'

	self.data = {}
	for source in self.sourcelist:
	    self.data[source] = tf.FixedLenFeature([], tf.string)
	    #TODO CHECK IF PREPROCESSING MAKES SENSE AT ALL HERE AND POSTPROCESSING SUFFICES
	    if source in self.preprocess:
		print('Preprocessing %s...' %s source)
		self.data[source] = self.preprocess[source](self.data[source])
	if self.decodelist is not None:
	    self.data['height'] = tf.FixedLenFeature([], tf.int64)
	    self.data['width'] = tf.FixedLenFeature([], tf.int64)
	    self.data['channels'] = tf.FixedLenFeature([], tf.int64)

	# setup tfrecords reader
	if self.usetensor:
	    self.reader = tf.TFRecordReader()
	    self.filename_queue = tf.train.string_input_producer([self.tfsource])
	    raise NotImplementedError('Check ToDos')
	else:
	    self.tfrec_ptr = tf.python_io.tf_record_iterator(path=self.tfsource)

	self.curr_batch_num = 0
	self.curr_epoch = 1

    def set_epoch_batch(self, epoch, batch_num):
	self.curr_epoch = epoch
	self.curr_batch_num = batch_num
	self.move_ptr_to(self.curr_batch_num)
	
    def move_ptr_to(self, batch_num):
	if self.usetensor:
	    self.reader = tf.TFRecordReader()
	    for i in range(batch_num):
		#TODO: Check for overflow
		reader.read_up_to(self.filename_queue, self.batch_size)
	else:
	    self.tfrec_ptr = tf.python_io.tf_record_iterator(path=self.tfsource)
	    for i in range(len(self.batch_size * batch_num)):
		try:
		    tfrec_ptr.next()
		except StopIteration:
		    raise IndexError('batch_num * batch_size > total records number: %d' % i)

    def __iter__(self):
	return self

    def next(self):
	if self.usetensor:
	    return self.get_next_batch_by_session()
	else:
	    return self.get_next_batch()

    def init_data(self):
	data = {}
        for source in self.sourcelist:
            data[source] = []
	return data

    def get_next_datum(self):
	try:
            return self.tfrec_ptr.next()
	# loop through data when at the end of the file
        except StopIteration:
            self.tfrec_ptr = tf.python_io.tf_record_iterator(path=self.tfsource)
            self.curr_batch_num = 0
            self.curr_epoch += 1
            return None

    def parse_and_append_datum(self, datum, data):
	example = tf.train.Example()
	example.ParseFromString(datum)
	if self.decodelist is not None:
	    height = int(example.features.feature['height'].int64_list.value[0])
            width = int(example.features.feature['width'].int64_list.value[0])
            channels = int(example.features.feature['channels'].int64_list.value[0])
	# parse and reshape data
        for source in self.sourcelist:
            if source in self.decodelist:
                serialized_data = (example.features.feature[source].bytes_list.value[0])
                data_1D = np.fromstring(serialized_data, dtype=np.uint8)
                data[source].append(data_1D.reshape( \
                        (self.batch_size, height, width, channels))
            else:
                data[source] = (example.features.feature[source].bytes_list.value[0])

	return data

    def get_next_batch(self):
	self.curr_batch_num += 1 
	data = self.init_data()
	# read and parse data 
	for i in range(self.batch_size):
	    datum = self.get_next_datum()
	    if datum is None:
		if self.pad:
		    # since we are looping over the file just try again
		    datum = self.get_next_datum()
		    if datum is None:
			raise StopIteration
		else:
		    break
	    else:
		data = self.parse_and_append_datum(datum, data)
	# convert to numpy arrays and postprocess
	for source in self.sourcelist:
            if source in self.decodelist:
		data[source] = np.array(data[source])
	    if source in self.postprocess:
		data[source] = self.postprocess[source](data[source])
	return data

    def get_next_batch_by_session(self):
	data = {}
	_, serialized_data = reader.read_up_to(self.filename_queue, self.batch_size)
	features = tf.parse_example(serialized_data, self.data)
	heights = tf.cast(features['height'], tf.int32)
	widths = tf.cast(features['width'], tf.int32)
	channels = tf.cast(features['channels'], tf.int32)
	shape = tf.pack([self.batch_size, heights[0], widths[0], channels[0]]))
	for source in self.sourcelist:
	    if source in self.decodelist:
		data[source] = tf.decode_raw(features[source], tf.uint8)
	    	data[source] = tf.reshape(data[source], shape)
	    else:
		data[source] = features[source]
	return data


class HDF5DataProvider(object):
    def __init__(self,
                 hdf5source,
                 sourcelist,
                 batch_size,
                 subslice=None,
                 mini_batch_size=None,
                 preprocess=None,
                 postprocess=None,
                 pad=False):

        """
        - hdf5source (str): path where hdf5 file resides
        - sourcelist (list of strs): list of keys in the hdf5file to use as source dataarrays
        - batch_size (int): size of batches to be returned
        - subslice (string, array of ints, callable):
             if str: name of key in hdf5file refering to array of indexes into the source dataarrays
             if array of ints: indexes into the source dataarrays
             if callable: function producing array of indexes into the source datarrays
           Regardless of how it's constructed, the provider subsets its returns to the only the indices
           specified in the subslice.
        - mini_batch_size (int):  Used only if subslice is specifiied, this sets the size of minibatches used
          when constructing one full batch within the subslice to return
        - preprocess (dict of callables): functions for preprocessing data in the datasources.  keys of this are subset
        - postprocess (dict of callables): functions for postprocess data.  Keys of this are subset of sourcelist.
        - pad (bool): whether to pad data returned if amount of data left to return is less then full batch size
        """
        self.hdf5source = hdf5source
        self.file = h5py.File(self.hdf5source, 'r')
        self.sourcelist = sourcelist
        self.subslice = subslice
        self.subsliceinds = None
        self.preprocess = {} if preprocess is None else preprocess
        self.postprocess = {} if postprocess is None else postprocess

        self.data = {}
        self.sizes = {}
        for source in self.sourcelist:
            self.data[source] = self.file[source]
            if source in self.preprocess:
                print('Preprocessing %s...' % source)
                self.data[source] = self.preprocess[source](self.data[source])

        for source in sourcelist:
            if self.subslice is None:
                self.sizes[source] = self.data[source].shape
            else:
                if self.subsliceinds is None:
                    if isinstance(self.subslice, str):
                        self.subsliceinds = self.file[self.subslice][:]
                    elif hasattr(self.subslice, '__call__'):
                        self.subsliceinds = self.subslice(self.file, self.sourcelist)
                    elif len(self.subslice) == self.data[source].shape[0]:
                        self.subsliceinds = self.subslice[:]
                    else:
                        self.subsliceinds = np.zeros(self.data[source].shape[0]).astype(np.bool)
                        self.subsliceinds[self.subslice] = True
                        self.subsliceinds = self.subsliceinds.nonzero()[0].astype(int)
                sz = self.data[source].shape
                self.sizes[source] = (self.subsliceinds.shape[0],) + sz[1:]
            if not hasattr(self, 'data_length'):
                self.data_length = self.sizes[source][0]
            assert self.sizes[source][0] == self.data_length, (self.sizes[source], self.data_length)

        self.batch_size = batch_size
        if mini_batch_size is None:
            mini_batch_size = self.batch_size
        self.mini_batch_size = mini_batch_size
        self.total_batches = (self.data_length - 1) // self.batch_size + 1
        self.curr_batch_num = 0
        self.curr_epoch = 1
        self.pad = pad

    def set_epoch_batch(self, epoch, batch_num):
        self.curr_epoch = epoch
        self.curr_batch_num = batch_num

    def get_next_batch(self):
        data = self.get_batch(self.curr_batch_num)
        self.increment_batch_num()
        return data

    def __iter__(self):
        return self

    def next(self):
        return self.get_next_batch()

    def increment_batch_num(self):
        m = self.total_batches
        if (self.curr_batch_num >= m - 1):
            self.curr_epoch += 1
        self.curr_batch_num = (self.curr_batch_num + 1) % m

    def get_batch(self, cbn):
        data = {}
        startv = cbn * self.batch_size
        endv = (cbn + 1) * self.batch_size
        if self.pad and endv > self.data_length:
            startv = self.data_length - self.batch_size
            endv = startv + self.batch_size
        sourcelist = self.sourcelist
        for source in sourcelist:
            data[source] = self.get_data(self.data[source], slice(startv, endv))
            if source in self.postprocess:
                data[source] = self.postprocess[source](data[source], self.file)
        return data

    def get_data(self, dsource, sliceval):
        if self.subslice is None:
            return dsource[sliceval]
        else:
            subslice_inds = self.subsliceinds[sliceval]
            mbs = self.mini_batch_size
            bn0 = subslice_inds.min() // mbs
            bn1 = subslice_inds.max() // mbs
            stims = []
            for _bn in range(bn0, bn1 + 1):
                _s = np.asarray(dsource[_bn * mbs: (_bn + 1) * mbs])
                new_inds = isin(np.arange(_bn * mbs, (_bn + 1) * mbs), subslice_inds)
                new_array = _s[new_inds]
                stims.append(new_array)
            stims = np.concatenate(stims)
            return stims


def get_unique_labels(larray):
    larray = larray[:]
    labels_unique = np.unique(larray)
    s = larray.argsort()
    cat_s = larray[s]
    ss = np.array([0] + ((cat_s[1:] != cat_s[:-1]).nonzero()[0] + 1).tolist() + [len(cat_s)])
    ssd = ss[1:] - ss[:-1]
    labels = np.repeat(np.arange(len(labels_unique)), ssd)
    larray = labels[perminverse(s)]
    return larray.astype(np.int64)


def perminverse(s):
    """
    Fast inverse of a (numpy) permutation.

    From yamutils
    """
    X = np.array(range(len(s)))
    X[s] = range(len(s))
    return X


def isin(X, Y):
    """
    Indices of elements in a numpy array that appear in another.
    Fast routine for determining indices of elements in numpy array `X` that
    appear in numpy array `Y`, returning a boolean array `Z` such that::
            Z[i] = X[i] in Y
    **Parameters**
            **X** :  numpy array
                    Numpy array to comapare to numpy array `Y`.  For each
                    element of `X`, ask if it is in `Y`.
            **Y** :  numpy array
                    Numpy array to which numpy array `X` is compared.  For each
                    element of `X`, ask if it is in `Y`.
    **Returns**
            **b** :  numpy array (bool)
                    Boolean numpy array, `len(b) = len(X)`.
    **See Also:**
            :func:`tabular.fast.recarrayisin`,
            :func:`tabular.fast.arraydifference`
    """
    if len(Y) > 0:
        T = Y.copy()
        T.sort()
        D = T.searchsorted(X)
        T = np.append(T, np.array([0]))
        W = (T[D] == X)
        if isinstance(W, bool):
            return np.zeros((len(X), ), bool)
        else:
            return (T[D] == X)
    else:
        return np.zeros((len(X), ), bool)


class Queue(object):
    """ A generic queue for reading data
        Built on top of https://indico.io/blog/tensorflow-data-input-part2-extensions/
    """

    def __init__(self,
                 data,
                 data_batch_size=None,
                 queue_type='fifo',
                 batch_size=256,
                 n_threads=4,
                 capacity=None,
                 seed=0):
        self.data_iter = iter(data)
        self.batch_size = batch_size
        self.n_threads = n_threads
        if capacity is None:
            self.capacity = self.n_threads * self.batch_size * 2
        else:
            self.capacity = capacity

        if data_batch_size is None:
            try:
                data_batch_size = self.data_iter.batch_size
            except KeyError:
                raise KeyError('Need to define data batch size; either pass it '
                               'to Queue constructor or have it defined in '
                               'data_iter.batch_size.')

        self.coord = tf.train.Coordinator()
        self._first_call = True
        self._first_batch = self.data_iter.next()
        self.nodes = {}
        dtypes = []
        shapes = []
        for key, value in self._first_batch.items():
            self.nodes[key] = tf.placeholder(value.dtype, shape=value.shape, name=key)
            dtypes.append(value.dtype)
            if data_batch_size > 1:
                shapes.append(value.shape[1:])
            else:
                shapes.append(value.shape)

        if queue_type == 'random':
            self.queue = tf.RandomShuffleQueue(capacity=self.capacity,
                                               min_after_dequeue=self.capacity // 2,
                                               dtypes=dtypes,
                                               shapes=shapes,
                                               names=self.nodes.keys(),
                                               seed=seed)
        elif queue_type == 'fifo':
            self.queue = tf.FIFOQueue(capacity=self.capacity,
                                      dtypes=dtypes,
                                      shapes=shapes,
                                      names=self.nodes.keys())
        elif queue_type == 'padding_fifo':
            self.queue = tf.PaddingFIFOQueue(capacity=self.capacity,
                                             dtypes=dtypes,
                                             shapes=shapes,
                                             names=self.nodes.keys())
        elif queue_type == 'priority':
            self.queue = tf.PriorityQueue(capacity=self.capacity,
                                          types=dtypes,
                                          shapes=shapes,
                                          names=self.nodes.keys())
        else:
            Exception('Queue type %s not recognized' % queue_type)

        if data_batch_size > 1:
            self.enqueue_op = self.queue.enqueue_many(self.nodes)
        else:
            self.enqueue_op = self.queue.enqueue(self.nodes)
        self.batch = self.queue.dequeue_many(batch_size)

    def __iter__(self):
        return self

    def next(self):
        if self._first_call:
            self._first_call = False
            return self._first_batch
        else:
            return self.data_iter.next()

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for batch in self:
            if not self.coord.should_stop():
                try:
                    feed_dict = {node: batch[name] for name, node in self.nodes.items()}
                    sess.run(self.enqueue_op, feed_dict=feed_dict)
                except tf.errors.CancelledError:
                    break
            else:
                break

    def start_threads(self, sess):
        """ Start background threads to feed queue """
        threads = []
        for n in range(self.n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        self.threads = threads

    def stop_threads(self, sess):
        self.coord.request_stop()
        close_op = self.queue.close(cancel_pending_enqueues=True)
        sess.run(close_op)


class MNIST(object):

    def __init__(self,
                 data_path=None,
                 group='train',
                 batch_size=100):
        """
        A specific reader for IamgeNet stored as a HDF5 file

        Kwargs:
            - data_path: path to imagenet data
            - group: train, validation, test
            - batch_size
        """
        self.batch_size = batch_size

        if data_path is None:
            data_path = '/tmp'
        data = read_data_sets(data_path)

        if group == 'train':
            self.data = data.train
        elif group == 'test':
            self.data = data.test
        elif group == 'validation':
            self.data = data.validation
        else:
            raise ValueError('MNIST data input "{}" not known'.format(group))

    def __iter__(self):
        return self

    def next(self):
        batch = self.data.next_batch(self.batch_size)
        feed_dict = {'images': batch[0], 'labels': batch[1].astype(np.int32)}
        return feed_dict


class ImageNet(HDF5DataProvider):

    N_TRAIN = 1281167
    N_VAL = 50000
    N_TRAIN_VAL = 50000

    def __init__(self,
                 data_path,
                 group='train',
                 batch_size=1,
                 crop_size=None,
                 *args,
                 **kwargs):
        """
        A specific reader for ImageNet stored as a HDF5 file

        Args:
            - data_path
                path to imagenet data
        Kwargs:
            - group (str, default: 'train')
                Which subset of the ImageNet set you want: train, val, train_val.
                The latter contains 50k images from the train set (50 per category),
                so that you can directly compare performance on the validation set
                to the performance on the train set to track overfitting.
            - batch_size (int, default: 1)
                Number of images to return when `next` is called. By default set
                to 1 since it is expected to be used with queues where reading one
                image at a time is ok.
            - crop_size (int or None, default: None)
                For center crop (crop_size x crop_size). If None, no cropping will occur.
            - *args, **kwargs
                Extra arguments for HDF5DataProvider
        """
        self.group = group
        images = group + '/images'
        labels = group + '/labels'
        super(ImageNet, self).__init__(
            data_path,
            [images, labels],
            batch_size=batch_size,
            postprocess={images: self.postproc_img, labels: self.postproc_labels},
            pad=True,
            *args, **kwargs)
        if crop_size is None:
            self.crop_size = 256
        else:
            self.crop_size = crop_size

    def postproc_img(self, ims, f):
        norm = ims.astype(np.float32) / 255
        off = np.random.randint(0, 256 - self.crop_size, size=2)
        images_batch = norm[:,
                            off[0]: off[0] + self.crop_size,
                            off[1]: off[1] + self.crop_size]
        return images_batch

    def postproc_labels(self, labels, f):
        return labels.astype(np.int32)

    def next(self):
        batch = super(ImageNet, self).next()
        feed_dict = {'images': np.squeeze(batch[self.group + '/images']),
                     'labels': np.squeeze(batch[self.group + '/labels'])}
        return feed_dict
