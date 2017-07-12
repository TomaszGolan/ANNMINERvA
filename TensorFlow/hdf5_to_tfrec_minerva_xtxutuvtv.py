"""
convert the hdf5 mnist file to tfrecords
"""
from __future__ import print_function
import h5py
import tensorflow as tf
import numpy as np


class minerva_hdf5_reader:
    """
    the `minerva_hdf5_reader` will return numpy ndarrays of data for given
    ranges. user should call `open()` and `close()` to start/finish.
    """
    def __init__(self, hdf5_file):
        self.file = hdf5_file
        self._f = None

    def open(self):
        self._f = h5py.File(self.file, 'r')

    def close(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

    def get_data(self, name, start_idx, stop_idx):
        return self._f[name][start_idx: stop_idx]

    def get_nevents(self):
        sizes = [self._f[d].shape[0] for d in self._f]
        if min(sizes) != max(sizes):
            raise ValueError("All dsets must have the same size!")
        return sizes[0]


def make_mnv_data_dict():
    # eventids are really uint64, planecodes are really uint16
    data_list = [
        ('eventids', tf.int64),
        ('hitimes-u', tf.float32),
        ('hitimes-v', tf.float32),
        ('hitimes-x', tf.float32),
        ('planecodes', tf.int16),
        ('segments', tf.uint8),
        ('zs', tf.float32)
    ]
    mnv_data = {}
    for datum in data_list:
        mnv_data[datum[0]] = {}
        mnv_data[datum[0]]['dtype'] = datum[1]
        mnv_data[datum[0]]['byte_data'] = None
    
    return mnv_data


def get_binary_data(reader, name, start_idx, stop_idx):
    """
    * reader - hdf5_reader
    * name of dset in the hdf5 file
    * indices
    returns byte data
    """
    dta = reader.get_data(name, start_idx, stop_idx)
    return dta.tobytes()


def write_to_tfrecord(data_dict, tfrecord_file):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    features_dict = {}
    for k in data_dict.keys():
        features_dict[k] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[data_dict[k]['byte_data']])
        )
    example = tf.train.Example(
        features=tf.train.Features(feature=features_dict)
    )
    writer.write(example.SerializeToString())
    writer.close()


def write_tfrecord(reader, data_dict, start_idx, stop_idx, tfrecord_file):
    for k in data_dict:
        data_dict[k]['byte_data'] = get_binary_data(
            reader, k, start_idx, stop_idx
        )
    write_to_tfrecord(data_dict, tfrecord_file)


def tfrecord_to_graph_ops_xtxutuvtv(filenames):
    def proces_hitimes(inp, shape):
        """
        *Note* - Minerva HDF5's are packed (N, C, H, W), so we must transpose
        them to (N, H, W, C) here.
        """
        hitimes = tf.decode_raw(inp, tf.float32)
        hitimes = tf.reshape(hitimes, shape)
        hitimes = tf.transpose(hitimes, [0, 2, 3, 1])
        return hitimes

    file_queue = tf.train.string_input_producer(
        filenames, name='file_queue'
    )
    reader = tf.TFRecordReader()
    _, tfrecord = reader.read(file_queue)

    tfrecord_features = tf.parse_single_example(
        tfrecord,
        features={
            'eventids': tf.FixedLenFeature([], tf.string),
            'hitimes-x': tf.FixedLenFeature([], tf.string),
            'hitimes-u': tf.FixedLenFeature([], tf.string),
            'hitimes-v': tf.FixedLenFeature([], tf.string),
            'planecodes': tf.FixedLenFeature([], tf.string),
            'segments': tf.FixedLenFeature([], tf.string),
            'zs': tf.FixedLenFeature([], tf.string),
        },
        name='data'
    )
    evtids = tf.decode_raw(tfrecord_features['eventids'], tf.int64)
    hitimesx = proces_hitimes(
        tfrecord_features['hitimes-x'], [-1, 2, 127, 50]
    )
    hitimesu = proces_hitimes(
        tfrecord_features['hitimes-u'], [-1, 2, 127, 25]
    )
    hitimesv = proces_hitimes(
        tfrecord_features['hitimes-v'], [-1, 2, 127, 25]
    )
    pcodes = tf.decode_raw(tfrecord_features['planecodes'], tf.int16)
    pcodes = tf.cast(pcodes, tf.int32)
    pcodes = tf.one_hot(indices=pcodes, depth=67, on_value=1, off_value=0)
    segs = tf.decode_raw(tfrecord_features['segments'], tf.uint8)
    segs = tf.cast(segs, tf.int32)
    segs = tf.one_hot(indices=segs, depth=11, on_value=1, off_value=0)
    zs = tf.decode_raw(tfrecord_features['zs'], tf.float32)
    return_dict = {}
    return_dict['eventids'] = evtids
    return_dict['hitimes-x'] = hitimesx
    return_dict['hitimes-u'] = hitimesu
    return_dict['hitimes-v'] = hitimesv
    return_dict['planecodes'] = pcodes
    return_dict['segments'] = segs
    return_dict['zs'] = zs
    return return_dict


def test_read_tfrecord(tfrecord_file):
    data_dict = tfrecord_to_graph_ops_xtxutuvtv([tfrecord_file])
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        evtids, hitsx, hitsu, hitsv, pcodes, segs, zs = sess.run([
            data_dict['eventids'],
            data_dict['hitimes-x'],
            data_dict['hitimes-u'],
            data_dict['hitimes-v'],
            data_dict['planecodes'],
            data_dict['segments'],
            data_dict['zs'],
        ])
        print('evtids shape =', evtids.shape)
        print('hitimes x shape =', hitsx.shape)
        print('hitimes u shape =', hitsu.shape)
        print('hitimes v shape =', hitsv.shape)
        print('planecodes shape =', pcodes.shape)
        print('  planecodes =', np.argmax(pcodes, axis=1))
        print('segments shape =', segs.shape)
        print('  segments =', np.argmax(segs, axis=1))
        print('zs shape =', zs.shape)
        coord.request_stop()
        coord.join(threads)


def write_all(hdf5_file, train_file, valid_file, test_file):
    m = minerva_hdf5_reader(hdf5_file)
    m.open()
    n_total = m.get_nevents()
    n_total = 100
    n_train = int(n_total * 0.88)
    n_valid = int(n_total * 0.07)
    n_test = n_total - n_train - n_valid
    print("{} total events".format(n_total))
    print("{} train events".format(n_train))
    print("{} valid events".format(n_valid))
    print("{} test events".format(n_test))
    data_dict = make_mnv_data_dict()
    # events included are [start, stop)
    print('creating train file...')
    write_tfrecord(m, data_dict, 0, n_train, train_file)
    print('creating valid file...')
    write_tfrecord(m, data_dict, n_train, n_train + n_valid, valid_file)
    print('creating test file...')
    write_tfrecord(m, data_dict, n_train + n_valid, n_total, test_file)
    m.close()


def read_all(train_file, valid_file, test_file):
    print('reading train file...')
    test_read_tfrecord(train_file)
    print('reading valid file...')
    test_read_tfrecord(valid_file)
    print('reading test file...')
    test_read_tfrecord(test_file)


if __name__ == '__main__':
    base_name = 'minosmatch_nukecczdefs_genallzwitht_pcodecap66'
    data_spec = '_127x50x25_xtxutuvtv_me1Amc_'
    subrun = '0000'
    
    hdf5_file = base_name + data_spec + subrun + '.hdf5'
    train_file = base_name + data_spec + subrun + '_train.tfrecord'
    valid_file = base_name + data_spec + subrun + '_valid.tfrecord'
    test_file = base_name + data_spec + subrun + '_test.tfrecord'

    write_all(hdf5_file, train_file, valid_file, test_file)
    read_all(train_file, valid_file, test_file)
