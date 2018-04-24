#!/usr/bin/env python
from collections import OrderedDict
import tensorflow as tf

from mnvtf.data_constants import EVENTIDS, PLANECODES, SEGMENTS, ZS
from mnvtf.data_constants import HITIMESU, HITIMESV, HITIMESX
from mnvtf.data_constants import QSQRD, WINV, XBJ, YBJ, CURRENT, INT_TYPE, TARGETZ
from mnvtf.data_constants import ESUM_CHGDKAONS, ESUM_CHGDPIONS, ESUM_HADMULTMEAS
from mnvtf.data_constants import ESUM_NEUTPIONS, ESUM_NEUTRONS
from mnvtf.data_constants import ESUM_OTHERS, ESUM_PROTONS
from mnvtf.data_constants import N_CHGDKAONS, N_CHGDPIONS, N_HADMULTMEAS
from mnvtf.data_constants import N_NEUTPIONS, N_NEUTRONS
from mnvtf.data_constants import N_OTHERS, N_PROTONS
from mnvtf.data_constants import ESUM_ELECTRONS, ESUM_MUONS, ESUM_TAUS
from mnvtf.data_constants import N_ELECTRONS, N_MUONS, N_TAUS


class MnvTFRecordReaderBase:
    """
    Minerva Data Reader for TFRecord files.

    name values are usually, e.g., 'train' or 'validation', etc.

    allowed compression values:
    * `tf.python_io.TFRecordCompressionType.ZLIB`
    * `tf.python_io.TFRecordCompressionType.GZIP`
    """
    def __init__(self, args_dict):
        self.filenames_list = args_dict['FILENAMES_LIST']
        self.batch_size = args_dict['BATCH_SIZE']
        self.name = args_dict['NAME']
        self.img_shp = args_dict['IMG_SHP']
        self.n_planecodes = args_dict['N_PLANECODES']
        self.data_format = args_dict['DATA_FORMAT']
        self.compression = args_dict['FILE_COMPRESSION']
        self.data_fields = None
        self._basic_int32_fields = set([
            CURRENT, INT_TYPE, TARGETZ,
            N_CHGDKAONS, N_CHGDPIONS, N_NEUTPIONS, N_NEUTRONS, N_OTHERS,
            N_PROTONS, N_ELECTRONS, N_MUONS, N_TAUS,
        ])
        self._basic_float32_fields = set([
            ZS, QSQRD, WINV, XBJ, YBJ,
            ESUM_CHGDKAONS, ESUM_CHGDPIONS, ESUM_HADMULTMEAS,
            ESUM_NEUTPIONS, ESUM_NEUTRONS, ESUM_OTHERS, ESUM_PROTONS,
            ESUM_ELECTRONS, ESUM_MUONS, ESUM_TAUS,
        ])

    def _process_hitimes(self, inp, shape):
        """ Start with a (N, C, H, W) structure, -> (N, H, W, C)? """
        tnsr = tf.reshape(tf.decode_raw(inp, tf.float32), shape)
        if self.data_format == 'NCHW':
            return tnsr
        elif self.data_format == 'NHWC':
            return tf.transpose(tnsr, [0, 2, 3, 1])
        else:
            raise ValueError('Invalid data format in data reader!')

    def _get_tfrecord_filequeue_and_reader(self, num_epochs):
        file_queue = tf.train.string_input_producer(
            self.filenames_list,
            name=self.name + '_file_queue',
            num_epochs=num_epochs
        )
        reader = tf.TFRecordReader(
            options=tf.python_io.TFRecordOptions(
                compression_type=self.compression
            )
        )
        _, tfrecord = reader.read(file_queue)
        return tfrecord

    def _decode_hitimesx(self, tfrecord_features):
        return self._process_hitimes(
            tfrecord_features[HITIMESX],
            [-1, self.img_shp[3], self.img_shp[0], self.img_shp[1]]
        )

    def _decode_hitimesu(self, tfrecord_features):
        return self._process_hitimes(
            tfrecord_features[HITIMESU],
            [-1, self.img_shp[3], self.img_shp[0], self.img_shp[2]]
        )

    def _decode_hitimesv(self, tfrecord_features):
        return self._process_hitimes(
            tfrecord_features[HITIMESV],
            [-1, self.img_shp[3], self.img_shp[0], self.img_shp[2]]
        )

    def _decode_basic(self, tfrecord_features, field, tf_dtype):
        return tf.decode_raw(tfrecord_features[field], tf_dtype)

    def _decode_onehot(
            self, tfrecord_features, field, depth, tf_dtype=tf.int32
    ):
        v = tf.decode_raw(tfrecord_features[field], tf_dtype)
        v = tf.one_hot(indices=v, depth=depth, on_value=1, off_value=0)
        return v

    def _decode_onehot_capped(
            self, tfrecord_features, field, depth, tf_dtype=tf.int32
    ):
        """depth becomes the cap value - should be ~ range(depth)"""
        v = tf.decode_raw(tfrecord_features[field], tf_dtype)
        m = (depth - 1) * tf.ones_like(v)
        z = tf.where(v < depth, x=v, y=m)
        z = tf.one_hot(indices=z, depth=depth, on_value=1, off_value=0)
        return z

    def _decode_tfrecord_feature(self, tfrecord_features, field):
        if field == EVENTIDS:
            return self._decode_basic(tfrecord_features, field, tf.int64)
        elif field == HITIMESU:
            return self._decode_hitimesu(tfrecord_features)
        elif field == HITIMESV:
            return self._decode_hitimesv(tfrecord_features)
        elif field == HITIMESX:
            return self._decode_hitimesx(tfrecord_features)
        elif field == PLANECODES:
            return self._decode_onehot(
                tfrecord_features, field, self.n_planecodes
            )
        elif field == SEGMENTS:
            return self._decode_onehot(tfrecord_features, field, 11, tf.uint8)
        elif field == N_HADMULTMEAS:
            # cap at 5 means we get {0, 1, 2, 3, 4+} hadrons
            return self._decode_onehot_capped(tfrecord_features, field, 5)
        elif field in self._basic_float32_fields:
            return self._decode_basic(tfrecord_features, field, tf.float32)
        elif field in self._basic_int32_fields:
            return self._decode_basic(tfrecord_features, field, tf.int32)

    def _tfrecord_to_graph_ops(self, num_epochs):
        od = OrderedDict()
        with tf.variable_scope(self.name + '_tfrec_to_graph_ops'):
            tfrecord = self._get_tfrecord_filequeue_and_reader(num_epochs)
            tfrecord_features = tf.parse_single_example(
                tfrecord,
                features=self._features_dict,
                name=self.name+'_data'
            )
            with tf.variable_scope(self.name + '_input_data'):
                for field in self.data_fields:
                    od[field] = self._decode_tfrecord_feature(
                        tfrecord_features, field
                    )
        return od

    def batch_generator(self, num_epochs=1):
        with tf.variable_scope(self.name+'_batch_gen'):
            od = self._tfrecord_to_graph_ops(num_epochs)
            capacity = 10 * self.batch_size
            batch_values = tf.train.batch(
                od.values(),
                batch_size=self.batch_size,
                capacity=capacity,
                enqueue_many=True,
                allow_smaller_final_batch=True,
                name=self.name+'_batch'
            )
        rd = OrderedDict(zip(od.keys(), batch_values))
        return rd

    def shuffle_batch_generator(self, num_epochs=1):
        with tf.variable_scope(self.name+'_shufflebatch_gen'):
            od = self._tfrecord_to_graph_ops(num_epochs)
            min_after_dequeue = 3 * self.batch_size
            capacity = 10 * self.batch_size
            batch_values = tf.train.shuffle_batch(
                od.values(),
                batch_size=self.batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                enqueue_many=True,
                allow_smaller_final_batch=True,
                name=self.name+'_shuffle_batch'
            )
        rd = OrderedDict(zip(od.keys(), batch_values))
        return rd


class MnvDataReaderImageST(MnvTFRecordReaderBase):
    """
    Minerva Data Reader for plain image "SpaceTime" data
    """
    def __init__(self, args_dict):
        """
        img_shp = (imgh, imgw_x, imgw_uv, img_depth)
        TODO - get the img depth into this call also...
        """
        MnvTFRecordReaderBase.__init__(self, args_dict)
        self.data_fields = sorted([
            EVENTIDS, HITIMESU, HITIMESV, HITIMESX
        ])
        self._features_dict = {
            f: tf.FixedLenFeature([], tf.string) for f in self.data_fields
        }


class MnvDataReaderVertexST(MnvTFRecordReaderBase):
    """
    Minerva Data Reader for (target) vertex-finder "SpaceTime" data
    """
    def __init__(self, args_dict):
        """
        img_shp = (imgh, imgw_x, imgw_uv, img_depth)
        TODO - get the img depth into this call also...
        """
        MnvTFRecordReaderBase.__init__(self, args_dict)
        self.data_fields = sorted([
            EVENTIDS, PLANECODES, SEGMENTS, ZS,
            HITIMESU, HITIMESV, HITIMESX
        ])
        self._features_dict = {
            f: tf.FixedLenFeature([], tf.string) for f in self.data_fields
        }


class MnvDataReaderHamultKineST(MnvTFRecordReaderBase):
    """
    Minerva Data Reader for hadmult-kine "SpaceTime" data
    """
    def __init__(self, args_dict):
        """
        img_shp = (imgh, imgw_x, imgw_uv, img_depth)
        TODO - get the img depth into this call also...
        """
        MnvTFRecordReaderBase.__init__(self, args_dict)
        self.data_fields = sorted([
            EVENTIDS, PLANECODES, SEGMENTS, ZS,
            HITIMESU, HITIMESV, HITIMESX,
            QSQRD, WINV, XBJ, YBJ, CURRENT, INT_TYPE, TARGETZ,
            ESUM_CHGDKAONS, ESUM_CHGDPIONS, ESUM_HADMULTMEAS,
            ESUM_NEUTPIONS, ESUM_NEUTRONS, ESUM_OTHERS, ESUM_PROTONS,
            N_CHGDKAONS, N_CHGDPIONS, N_HADMULTMEAS,
            N_NEUTPIONS, N_NEUTRONS, N_OTHERS, N_PROTONS,
            ESUM_ELECTRONS, ESUM_MUONS, ESUM_TAUS,
            N_ELECTRONS, N_MUONS, N_TAUS,
        ])
        self._features_dict = {
            f: tf.FixedLenFeature([], tf.string) for f in self.data_fields
        }

class MnvDataReaderSegmentST(MnvTFRecordReaderBase):
    """
    Minerva Data Reader for segmentation "SpaceTime" data
    """
    def __init__(self, args_dict):
        """
        img_shp = (imgh, imgw_x, imgw_uv, img_depth)
        TODO - get the img depth into this call also...
        """
        MnvTFRecordReaderBase.__init__(self, args_dict)
        self.data_fields = sorted([
            EVENTIDS,
            HITIMESU, HITIMESV, HITIMESX,
        ])
        self._features_dict = {
            f: tf.FixedLenFeature([], tf.string) for f in self.data_fields
        }

