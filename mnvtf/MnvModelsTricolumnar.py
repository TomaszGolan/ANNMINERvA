"""
MINERvA Tri-columnar "spacetime" (energy+time tensors) epsilon network
"""
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier_init

import logging

from six.moves import range

LOGGER = logging.getLogger(__name__)


def make_default_convpooldict(
        img_depth=1, data_format='NHWC', use_batch_norm=False
):
    """
    return the default network structure dictionary

    conv kernel shape is [filt_h, filt_w, in_nch, out_nch]
    pool kernel shape is [N, H, W, C] for 'NHWC', etc.
    """
    convpooldict = {}

    if data_format == 'NHWC':
        pool_ksize = [1, 2, 1, 1]
        pool_strides = [1, 2, 1, 1]
    elif data_format == 'NCHW':
        pool_ksize = [1, 1, 2, 1]
        pool_strides = [1, 1, 2, 1]
    else:
        raise Exception('Invalid data format!')

    convpooldict['pool_ksize'] = pool_ksize
    convpooldict['pool_strides'] = pool_strides
    convpooldict['use_batch_norm'] = use_batch_norm

    # assume 127x(N) images
    convpooldict_x = {}
    convpooldict_x['conv1'] = {}
    convpooldict_x['conv2'] = {}
    convpooldict_x['conv3'] = {}
    convpooldict_x['conv4'] = {}
    convpooldict_x['conv1']['kernels'] = [8, 3, img_depth, 12]
    convpooldict_x['conv1']['biases'] = [12]
    # after 8x3 filters -> 120x(N-2) image, then maxpool -> 60x(N-2)
    convpooldict_x['conv2']['kernels'] = [7, 3, 12, 20]
    convpooldict_x['conv2']['biases'] = [20]
    # after 7x3 filters -> 54x(N-4) image, then maxpool -> 27x(N-4)
    convpooldict_x['conv3']['kernels'] = [6, 3, 20, 28]
    convpooldict_x['conv3']['biases'] = [28]
    # after 6x3 filters -> 22x(N-6) image, then maxpool -> 11x(N-6)
    convpooldict_x['conv4']['kernels'] = [6, 3, 28, 36]
    convpooldict_x['conv4']['biases'] = [36]
    # after 6x3 filters -> 6x(N-6) image, then maxpool -> 3x(N-6)
    convpooldict['x'] = convpooldict_x

    # assume 127x(N) images
    convpooldict_u = {}
    convpooldict_u['conv1'] = {}
    convpooldict_u['conv2'] = {}
    convpooldict_u['conv3'] = {}
    convpooldict_u['conv4'] = {}
    convpooldict_u['conv1']['kernels'] = [8, 5, img_depth, 12]
    convpooldict_u['conv1']['biases'] = [12]
    # after 8x3 filters -> 120x(N-4) image, then maxpool -> 60x(N-4)
    convpooldict_u['conv2']['kernels'] = [7, 3, 12, 20]
    convpooldict_u['conv2']['biases'] = [20]
    # after 7x3 filters -> 54x(N-6) image, then maxpool -> 27x(N-6)
    convpooldict_u['conv3']['kernels'] = [6, 3, 20, 28]
    convpooldict_u['conv3']['biases'] = [28]
    # after 6x3 filters -> 22x(N-8) image, then maxpool -> 11x(N-8)
    convpooldict_u['conv4']['kernels'] = [6, 3, 28, 36]
    convpooldict_u['conv4']['biases'] = [36]
    # after 6x3 filters -> 6x(N-10) image, then maxpool -> 3x(N-10)
    convpooldict['u'] = convpooldict_u

    # assume 127x(N) images
    convpooldict_v = {}
    convpooldict_v['conv1'] = {}
    convpooldict_v['conv2'] = {}
    convpooldict_v['conv3'] = {}
    convpooldict_v['conv4'] = {}
    convpooldict_v['conv1']['kernels'] = [8, 5, img_depth, 12]
    convpooldict_v['conv1']['biases'] = [12]
    # after 8x3 filters -> 120x(N-4) image, then maxpool -> 60x(N-4)
    convpooldict_v['conv2']['kernels'] = [7, 3, 12, 20]
    convpooldict_v['conv2']['biases'] = [20]
    # after 7x3 filters -> 54x(N-6) image, then maxpool -> 27x(N-6)
    convpooldict_v['conv3']['kernels'] = [6, 3, 20, 28]
    convpooldict_v['conv3']['biases'] = [28]
    # after 6x3 filters -> 22x(N-8) image, then maxpool -> 11x(N-8)
    convpooldict_v['conv4']['kernels'] = [6, 3, 28, 36]
    convpooldict_v['conv4']['biases'] = [36]
    # after 6x3 filters -> 6x(N-10) image, then maxpool -> 3x(N-10)
    convpooldict['v'] = convpooldict_v

    convpooldict['nfeat_dense_tower'] = 196
    convpooldict['nfeat_concat_dense'] = 98

    convpooldict['regularizer'] = tf.contrib.layers.l2_regularizer(
        scale=0.0001
    )

    return convpooldict


class LayerCreator:
    def __init__(
            self, regularization_strategy='l2', regularization_scale=0.0001,
            use_batch_norm=False, data_format='NHWC', padding='VALID'
    ):
        if regularization_strategy == 'l2':
            self.reg = tf.contrib.layers.l2_regularizer(
                scale=regularization_scale
            )
        else:
            raise NotImplementedError(
                'Regularization strategy ' + regularization_strategy + ' \
                is not implemented yet.'
            )
        self.use_batch_norm = use_batch_norm
        self.batch_norm_decay = 0.999
        self.is_training = None
        self.data_format = data_format
        self.padding = padding  # TODO - layer by layer option?
        # self.dropout_config??

    def set_is_training_placeholder(self, is_training):
        self.is_training = is_training

    def make_wbkernels(
            self, name, shp=None, initializer=xavier_init(uniform=False),
    ):
        """ make weights, biases, kernels """
        return tf.get_variable(
            name, shp, initializer=initializer, regularizer=self.reg
        )

    def make_fc_layer(
            self, inp_lyr, name_fc_lyr,
            name_w, shp_w, name_b=None, shp_b=None,
            initializer=xavier_init(uniform=False)
    ):
        """ TODO - regularize batch norm params? """
        W = self.make_wbkernels(name_w, shp_w, initializer=initializer)
        b = self.make_wbkernels(
            name_b, shp_b, initializer=tf.zeros_initializer()
        )
        fc_lyr = tf.nn.bias_add(
            tf.matmul(inp_lyr, W, name=name_fc_lyr+'_matmul'), b,
            data_format=self.data_format, name=name_fc_lyr,
        )
        if self.use_batch_norm:
            fc_lyr = tf.contrib.layers.batch_norm(
                fc_lyr, decay=self.batch_norm_decay, center=True, scale=True,
                data_format=self.data_format, is_training=self.is_training
            )
        return fc_lyr
            
    def make_active_fc_layer(
            self, inp_lyr, name_fc_lyr,
            name_w, shp_w, name_b=None, shp_b=None,
            act=tf.nn.relu, initializer=xavier_init(uniform=False)
    ):
        return act(self.make_fc_layer(
            inp_lyr, name_fc_lyr, name_w, shp_w, name_b, shp_b,
            initializer=initializer
        ), name=name_fc_lyr+'_act')

    def make_active_conv(
            self, input_lyr, name, kernels,
            biases=None, act=tf.nn.relu, strides=[1, 1, 1, 1]
    ):
        """ TODO - regularize batch norm params? biases? """
        conv = tf.nn.conv2d(
            input_lyr, kernels, strides=strides,
            padding=self.padding, data_format=self.data_format,
            name=name
        )
        if self.use_batch_norm:
            # TODO - test `activation_fun` argument
            return act(
                tf.contrib.layers.batch_norm(
                    tf.nn.bias_add(
                        conv, biases, data_format=self.data_format,
                        name=name+'_plus_biases'
                    ), decay=self.batch_norm_decay,
                    center=True, scale=True,
                    data_format=self.data_format, is_training=self.is_training
                ),
                name=name+'_act'
            )
        else:
            return act(tf.nn.bias_add(
                conv, biases, data_format=self.data_format, name=name+'_act'
            ))

    def make_pool(
            self, input_lyr, name, ksize, strides, padding=None
    ):
        padding = padding or self.padding
        pool = tf.nn.max_pool(
            input_lyr, ksize=ksize, strides=strides,
            padding=padding, data_format=self.data_format, name=name
        )
        return pool


def create_train_valid_summaries(
        loss, accuracy=None, reg_loss=None, do_valid=True
):
    base_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    train_summaries, valid_summaries = [], []
    train_summary_op, valid_summary_op = None, None
    with tf.name_scope('summaries/train'):
        train_loss = tf.summary.scalar('loss', loss)
        train_histo_loss = tf.summary.histogram(
            'histogram_loss', loss
        )
        train_summaries.extend([train_loss, train_histo_loss])
        if reg_loss is not None:
            train_reg_loss = tf.summary.scalar('reg_loss', reg_loss)
            train_summaries.append(train_reg_loss)
        if accuracy is not None:
            train_accuracy = tf.summary.scalar('accuracy', accuracy)
            train_summaries.append(train_accuracy)
        train_summaries.extend(base_summaries)
        train_summary_op = tf.summary.merge(train_summaries)
    if do_valid:
        with tf.name_scope('summaries/valid'):
            valid_loss = tf.summary.scalar('loss', loss)
            valid_histo_loss = tf.summary.histogram(
                'histogram_loss', loss
            )
            valid_summaries.extend([valid_loss, valid_histo_loss])
            if reg_loss is not None:
                valid_reg_loss = tf.summary.scalar('reg_loss', reg_loss)
                valid_summaries.append(valid_reg_loss)
            if accuracy is not None:
                valid_accuracy = tf.summary.scalar('accuracy', accuracy)
                valid_summaries.append(valid_accuracy)
            valid_summaries.extend(base_summaries)
            valid_summary_op = tf.summary.merge(valid_summaries)
    return train_summary_op, valid_summary_op


def compute_categorical_loss_and_accuracy(logits, targets):
    """return total loss, reg loss (subset of total), and accuracy"""
    with tf.variable_scope('loss'):
        regularization_losses = sum(
            tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES
            )
        )
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=targets
            ),
            axis=0,
            name='loss'
        ) + regularization_losses
        preds = tf.nn.softmax(logits, name='preds')
        correct_preds = tf.equal(
            tf.argmax(preds, 1), tf.argmax(targets, 1),
            name='correct_preds'
        )
        accuracy = tf.divide(
            tf.reduce_sum(tf.cast(correct_preds, tf.float32)),
            tf.cast(tf.shape(targets)[0], tf.float32),
            name='accuracy'
        )
    return loss, regularization_losses, accuracy


def make_standard_placeholders():
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    global_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name='global_step'
    )
    is_training = tf.placeholder(tf.bool, name='is_training')
    return dropout_keep_prob, global_step, is_training
    

class TriColSTEpsilon:
    """
    Tri-Columnar SpaceTime Epsilon
    """
    _allowed_strategies = ['Adam', 'AdaGrad']
    
    def __init__(self, n_classes, data_format='NHWC', use_batch_norm=False):
        """ note, 'NCHW' is only supported on GPUs """
        self.n_classes = n_classes
        self.loss = None
        self.logits = None
        self.dropout_keep_prob = None
        self.global_step = None
        self.is_training = None
        self.padding = 'VALID'
        self.data_format = data_format
        self.layer_creator = LayerCreator(
            'l2', 0.0001, use_batch_norm, self.data_format, self.padding
        )

    def _build_network(self, features_list, kbd):
        """
        _build_network does a bit more than just build the network; it also
        handles some initialization that is tricky in the init function in
        terms of belonging to the same graph as what is actually used.

        features_list[0] == X, [1] == U, [2] == V;
        kbd = kernels-biases-dict (convpooldict)
        """
        LOGGER.info('Building network from structure: %s' % str(kbd))
        self.dropout_keep_prob, self.global_step, self.is_training = \
            make_standard_placeholders()
        lc = self.layer_creator
        lc.set_is_training_placeholder(self.is_training)

        with tf.variable_scope('input_images'):
            self.X_img = tf.cast(features_list[0], tf.float32)
            self.U_img = tf.cast(features_list[1], tf.float32)
            self.V_img = tf.cast(features_list[2], tf.float32)

        def make_convolutional_tower(view, input_layer, kbd, n_layers=4):
            """
            input_layer == e.g., self.X_img, etc. corresponding to 'view'
            """
            inp_lyr = None
            out_lyr = None

            # scope the tower
            twr = view + '_tower'
            with tf.variable_scope(twr):

                for i in range(n_layers):
                    layer = str(i + 1)
                    if layer == '1':
                        inp_lyr = input_layer
                    else:
                        inp_lyr = out_lyr

                    # scope the convolutional layer
                    nm = 'conv' + layer
                    with tf.variable_scope(nm):
                        k = lc.make_wbkernels(
                            'kernels', kbd[view][nm]['kernels']
                        )
                        b = lc.make_wbkernels(
                            'biases', kbd[view][nm]['biases']
                        )
                        conv = lc.make_active_conv(inp_lyr, nm+'_conv', k, b)

                    # scope the pooling layer
                    scope_name = 'pool' + layer
                    with tf.variable_scope(scope_name):
                        out_lyr = lc.make_pool(
                            conv, scope_name+'_pool',
                            kbd['pool_ksize'], kbd['pool_strides']
                        )

                # reshape pool/out_lyr to 2 dimensional
                out_lyr_shp = out_lyr.shape.as_list()
                nfeat_tower = out_lyr_shp[1] * out_lyr_shp[2] * out_lyr_shp[3]
                out_lyr = tf.reshape(out_lyr, [-1, nfeat_tower])

                # make final active dense layer
                with tf.variable_scope('fully_connected'):
                    fc = lc.make_active_fc_layer(
                        out_lyr, 'fc_relu',
                        'dense_weights',
                        [nfeat_tower, kbd['nfeat_dense_tower']],
                        'dense_biases',
                        [kbd['nfeat_dense_tower']]
                    )
                    fc = tf.nn.dropout(
                        fc, self.dropout_keep_prob, name='relu_dropout'
                    )

            return fc

        with tf.variable_scope('model'):
            out_x = make_convolutional_tower('x', self.X_img, kbd)
            out_u = make_convolutional_tower('u', self.U_img, kbd)
            out_v = make_convolutional_tower('v', self.V_img, kbd)

            # next, concat, then 'final' fc...
            with tf.variable_scope('fully_connected'):
                tower_joined = tf.concat(
                    [out_x, out_u, out_v], axis=1, name='concat'
                )
                joined_shp = tower_joined.shape.as_list()
                nfeatures_joined = joined_shp[1]
                fc_lyr = lc.make_active_fc_layer(
                    tower_joined, 'fc_relu',
                    'weights', [nfeatures_joined, kbd['nfeat_concat_dense']],
                    'biases', [kbd['nfeat_concat_dense']]
                )
                self.fc = tf.nn.dropout(
                    fc_lyr, self.dropout_keep_prob, name='relu_dropout'
                )

            with tf.variable_scope('softmax_linear'):
                self.weights_softmax = lc.make_wbkernels(
                    'weights', [kbd['nfeat_concat_dense'], self.n_classes]
                )
                self.biases_softmax = lc.make_wbkernels(
                    'biases', [self.n_classes],
                    initializer=tf.zeros_initializer()
                )
                self.logits = tf.nn.bias_add(
                    tf.matmul(self.fc, self.weights_softmax),
                    self.biases_softmax,
                    data_format=self.data_format,
                    name='logits'
                )

    def _set_targets(self, targets):
        with tf.variable_scope('targets'):
            self.targets = tf.cast(targets, tf.float32)

    def _define_loss(self):
        self.loss, self.regularization_losses, self.accuracy = \
            compute_categorical_loss_and_accuracy(self.logits, self.targets)

    def _define_train_op(self, learning_rate, strategy):
        LOGGER.info('Building train op with learning_rate = %f' %
                    learning_rate)
        if strategy in self._allowed_strategies:
            with tf.variable_scope('training'):
                # need update_ops for batch normalization
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    if strategy == 'Adam':
                        self.optimizer = tf.train.AdamOptimizer(
                            learning_rate=learning_rate
                        ).minimize(self.loss, global_step=self.global_step)
                    elif strategy == 'AdaGrad':
                        self.optimizer = tf.train.MomentumOptimizer(
                            learning_rate=learning_rate,
                            momentum=0.9, use_nesterov=True
                        ).minimize(self.loss, global_step=self.global_step)
        else:
            raise ValueError('Invalid training strategy choice!')

    def _create_summaries(self):
        # assume we built the readers before the model...
        self.train_summary_op, self.valid_summary_op =  \
            create_train_valid_summaries(
                self.loss, self.accuracy, self.regularization_losses
            )

    def prepare_for_inference(self, features, kbd):
        """ kbd == kernels_biases_dict (convpooldict) """
        self._build_network(features, kbd)

    def prepare_for_training(
            self, targets, learning_rate=0.001, strategy='Adam'
    ):
        """ prep the train op and compute loss, plus summaries """
        self._set_targets(targets)
        self._define_loss()
        self._define_train_op(learning_rate, strategy)
        self._create_summaries()

    def prepare_for_loss_computation(self, targets):
        """ compute the loss and prepare summaries (good for testing) """
        self._set_targets(targets)
        self._define_loss()
        self._create_summaries()

    def get_output_nodes(self):
        """ list of output nodes for graph freezing """
        return ["model/softmax_linear/logits"]


def test():
    tf.reset_default_graph()
    img_depth = 2
    Xshp = [None, 127, 50, img_depth]
    UVshp = [None, 127, 25, img_depth]
    X = tf.placeholder(tf.float32, shape=Xshp, name='X')
    U = tf.placeholder(tf.float32, shape=UVshp, name='U')
    V = tf.placeholder(tf.float32, shape=UVshp, name='V')
    targ = tf.placeholder(tf.float32, shape=[None, 11], name='targ')
    f = [X, U, V]
    d = make_default_convpooldict(img_depth=img_depth)
    t = TriColSTEpsilon(11)
    t.prepare_for_inference(f, d)
    t.prepare_for_training(targ)


if __name__ == '__main__':
    test()