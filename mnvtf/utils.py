#!/usr/bin/env python
import tensorflow as tf
import logging
import os
import shutil
import gzip

from mnvtf.data_readers import MnvDataReaderImageST
from mnvtf.data_readers import MnvDataReaderVertexST
from mnvtf.data_readers import MnvDataReaderHamultKineST
from mnvtf.data_readers import MnvDataReaderSegmentST
from mnvtf.data_constants import HADMULTKINE_GROUPS_DICT, HADMULTKINE_TYPE
from mnvtf.data_constants import VTXFINDING_GROUPS_DICT, VTXFINDING_TYPE
from mnvtf.data_constants import IMGING_GROUPS_DICT, IMGING_TYPE
from mnvtf.data_constants import SEGMENTATION_GROUPS_DICT, SEGMENTATION_TYPE
from mnvtf.models_tricolumnar import TriColSTEpsilon
from mnvtf.models_tricolumnar import make_default_convpooldict
from mnvtf.models_menagerie import make_menndl_633167

LOGGER = logging.getLogger(__name__)

ZLIB_COMP = tf.python_io.TFRecordCompressionType.ZLIB
GZIP_COMP = tf.python_io.TFRecordCompressionType.GZIP
NONE_COMP = tf.python_io.TFRecordCompressionType.NONE


def make_data_reader_dict(
        filenames_list=None,
        batch_size=128,
        name='reader',
        compression=None,
        img_shp=(127, 50, 25, 2),
        data_format='NHWC',
        n_planecodes=67
):
    """
    NOTE: `img_shp` is not a regular TF 4-tensor.
    img_shp = (height, width_x, widht_uv, depth)
    """
    data_reader_dict = {}
    data_reader_dict['FILENAMES_LIST'] = filenames_list
    data_reader_dict['BATCH_SIZE'] = batch_size
    data_reader_dict['NAME'] = name
    if compression is None:
        data_reader_dict['FILE_COMPRESSION'] = NONE_COMP
    elif compression == 'zz':
        data_reader_dict['FILE_COMPRESSION'] = ZLIB_COMP
    elif compression == 'gz':
        data_reader_dict['FILE_COMPRESSION'] = GZIP_COMP
    else:
        msg = 'Invalid compression type in utils!'
        LOGGER.error(msg)
        raise ValueError(msg)
    data_reader_dict['IMG_SHP'] = img_shp
    data_reader_dict['DATA_FORMAT'] = data_format
    data_reader_dict['N_PLANECODES'] = n_planecodes
    return data_reader_dict


def make_run_params_dict(mnv_type='st_epsilon', tf_flags=None):
    run_params_dict = {}
    run_params_dict['DATA_READER_CLASS'] = None
    run_params_dict['MODEL_DIR'] = tf_flags.model_dir \
        if tf_flags else '/tmp/model'
    run_params_dict['LOAD_SAVED_MODEL'] = True
    run_params_dict['SAVE_EVRY_N_BATCHES'] = tf_flags.save_every_n_batch \
        if tf_flags else 500
    run_params_dict['BE_VERBOSE'] = tf_flags.be_verbose \
        if tf_flags else False
    run_params_dict['PREDICTION_STORE_NAME'] = tf_flags.pred_store_name \
        if tf_flags else 'preds'
    run_params_dict['CONFIG_PROTO'] = None
    if tf_flags and tf_flags.do_log_devices:
        run_params_dict['CONFIG_PROTO'] = tf.ConfigProto(
            log_device_placement=True
        )
    run_params_dict['PRED_STORE_USE_DB'] = False
    return run_params_dict


def make_feature_targ_dict(mnv_type='st_epsilon', tf_flags=None):
    feature_targ_dict = {}
    if mnv_type == 'st_epsilon':
        feature_targ_dict['TARGETS_LABEL'] = tf_flags.targets_label \
            if tf_flags else None
        feature_targ_dict['BUILD_KBD_FUNCTION'] = None
        feature_targ_dict['IMG_DEPTH'] = tf_flags.img_depth \
            if tf_flags else 1
    return feature_targ_dict


def make_train_params_dict(mnv_type='st_epsilon', tf_flags=None):
    train_params_dict = {}
    train_params_dict['LEARNING_RATE'] = tf_flags.learning_rate \
        if tf_flags else 0.001
    train_params_dict['NUM_EPOCHS'] = tf_flags.num_epochs \
        if tf_flags else 1
    train_params_dict['MOMENTUM'] = 0.9
    train_params_dict['STRATEGY'] = tf_flags.strategy \
        if tf_flags else "Adam"
    train_params_dict['DROPOUT_KEEP_PROB'] = 0.5
    return train_params_dict


def get_logging_level(log_level):
    logging_level = logging.INFO
    if log_level == 'DEBUG':
        logging_level = logging.DEBUG
    elif log_level == 'INFO':
        logging_level = logging.INFO
    elif log_level == 'WARNING':
        logging_level = logging.WARNING
    elif log_level == 'ERROR':
        logging_level = logging.ERROR
    elif log_level == 'CRITICAL':
        logging_level = logging.CRITICAL
    else:
        print('Unknown or unset logging level. Using INFO')

    return logging_level


def get_trainvalidtest_file_lists(data_dir_str, file_root_str, compression):
    """
    Assume we are looking for three sets of files in the form of
    TFRecords, with groups of files with extensions like
    `..._train.tfrecord`, `..._valid.tfrecord`, and
    `..._test.tfrecord` (possibly with .zz or .gz compression
    extension markers).
    """
    import glob
    comp_ext = compression if compression == '' else '.' + compression
    train_list = []
    valid_list = []
    test_list = []
    for data_dir in data_dir_str.split(','):
        for file_root in file_root_str.split(','):
            train_list.extend(
                glob.glob(data_dir + '/' + file_root +
                          '*_train.tfrecord' + comp_ext)
            )
            valid_list.extend(
                glob.glob(data_dir + '/' + file_root +
                          '*_valid.tfrecord' + comp_ext)
            )
            test_list.extend(
                glob.glob(data_dir + '/' + file_root +
                          '*_test.tfrecord' + comp_ext)
            )
    for t, l in zip(['training', 'validation', 'test'],
                    [train_list, valid_list, test_list]):
        LOGGER.debug('{} file list ='.format(t))
        for filename in l:
            LOGGER.debug('  {}'.format(filename))
    if len(train_list) == 0 and \
       len(valid_list) == 0 and \
       len(test_list) == 0:
        LOGGER.error('No files found at specified path!')
        return None, None, None
    return train_list, valid_list, test_list


def get_number_of_trainable_parameters():
    """ use default graph """
    # https://stackoverflow.com/questions/38160940/ ...
    LOGGER.debug('Now compute total number of trainable params...')
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        name = variable.name
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        LOGGER.debug(' layer name = {}, shape = {}, n_params = {}'.format(
            name, shape, variable_parameters
        ))
        total_parameters += variable_parameters
    LOGGER.debug('Total parameters = %d' % total_parameters)
    return total_parameters


def gz_compress(out_file):
    gzfile = out_file + '.gz'
    with open(out_file, 'rb') as f_in, gzip.open(gzfile, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    if os.path.isfile(gzfile) and (os.stat(gzfile).st_size > 0):
        os.remove(out_file)
    else:
        raise IOError('Compressed file not produced!')


def get_groups_list(hdf5_type):
    """
    get the list of HDF5 'groups' for a given ROOT->HDF5 conversion
    """
    if hdf5_type == VTXFINDING_TYPE:
        return VTXFINDING_GROUPS_DICT.keys()
    elif hdf5_type == HADMULTKINE_TYPE:
        return HADMULTKINE_GROUPS_DICT.keys()
    elif hdf5_type == IMGING_TYPE:
        return IMGING_GROUPS_DICT.keys()
    elif hdf5_type == SEGMENTATION_TYPE:
        return SEGMENTATION_GROUPS_DICT.keys()
    else:
        raise ValueError('Unknown HDF5 grouping type {}!'.format(hdf5_type))


def get_reader_class(data_file_type):
    """ get TFRecord reader class """
    if data_file_type == VTXFINDING_TYPE:
        return MnvDataReaderVertexST
    elif data_file_type == HADMULTKINE_TYPE:
        return MnvDataReaderHamultKineST
    elif data_file_type == IMGING_TYPE:
        return MnvDataReaderImageST
    elif data_file_type == SEGMENTATION_TYPE:
        return MnvDataReaderSegmentST
    else:
        raise ValueError('Unknown TFRec data file type!')


def get_network_model_class(model_type):
    if model_type == 'TriColSTEpsilon':
        return TriColSTEpsilon
    else:
        raise ValueError('Unknown NN model type!')


def get_kbd_function(network_type):
    if network_type == 'default':
        return make_default_convpooldict
    elif network_type == 'menndl_633167':
        return make_menndl_633167
    else:
        raise ValueError('Unknown NN network type!')


def decode_eventid(eventid):
    """
    assume "standard" encoding
    """
    evtid = str(eventid)
    phys_evt = evtid[-2:]
    evtid = evtid[:-2]
    gate = evtid[-4:]
    evtid = evtid[:-4]
    subrun = evtid[-4:]
    evtid = evtid[:-4]
    run = evtid
    return (run, subrun, gate, phys_evt)


def encode_eventid(run, subrun, gate, phys_evt):
    run = '{0:06d}'.format(int(run))
    subrun = '{0:04d}'.format(int(subrun))
    gate = '{0:04d}'.format(int(gate))
    phys_evt = '{0:02d}'.format(int(phys_evt))
    return run + subrun + gate + phys_evt


def freeze_graph(
        model_dir, output_nodes_list, output_graph_name='frozen_model.pb'
):
    """
    reduce a saved model and metadata down to a deployable file; following
    https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

    output_node_names = e.g., 'softmax_linear/logits'
    """
    from tensorflow.python.framework import graph_util

    LOGGER.info('Attempting to freeze graph at {}'.format(model_dir))
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    if input_checkpoint is None:
        LOGGER.error('Cannot load checkpoint at {}'.format(model_dir))
        return None

    # specify the output graph
    absolute_model_dir = '/'.join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + '/' + output_graph_name

    # import the meta graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)
    # get the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # convert variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_nodes_list
        )
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        LOGGER.info('Froze graph with {} ops'.format(
            len(output_graph_def.node)
        ))

    return output_graph
