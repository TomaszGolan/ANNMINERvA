#!/usr/bin/env python
"""
Run TF
"""
from __future__ import print_function
import os
import time
import logging

import tensorflow as tf

# TODO - shouldn't need to import models here... pass them in
from MnvModelsTricolumnar import TriColSTEpsilon
from MnvModelsTricolumnar import make_default_convpooldict
from MnvDataReaders import MnvDataReaderVertexST

logger = logging.getLogger(__name__)


class MnvTFRunnerCategorical:
    """
    Minerva runner class for categorical classification
    (not sure we need to make this distinction here)
    """
    def __init__(
            self,
            model,
            run_params_dict,
            feature_targ_dict=dict(),
            train_params_dict=dict(),
            img_params_dict=dict(),
    ):
        try:
            self.train_file_list = run_params_dict['TRAIN_FILE_LIST']
            self.valid_file_list = run_params_dict['VALID_FILE_LIST']
            self.test_file_list = run_params_dict['TEST_FILE_LIST']
            self.file_compression = run_params_dict['COMPRESSION']
            self.save_model_directory = run_params_dict['MODEL_DIR']
            self.load_saved_model = run_params_dict['LOAD_SAVED_MODEL']
            self.save_freq = run_params_dict['SAVE_EVRY_N_EVTS']
            # TODO - debug print and verbose should control logger levels
            self.debug_print = run_params_dict['DEBUG_PRINT']
            self.be_verbose = run_params_dict['BE_VERBOSE']
            self.write_db = run_params_dict['WRITE_DB']
        except KeyError as e:
            print(e)

        self.features = feature_targ_dict.get(
            'FEATURE_STR_DICT',
            dict([('x', 'hitimes-x'),
                  ('u', 'hitimes-u'),
                  ('v', 'hitimes-v')])
        )
        self.targets_label = feature_targ_dict.get(
            'TARGETS_LABEL', 'segments'
        )

        self.learning_rate = train_params_dict.get('LEARNING_RATE', 0.001)
        self.batch_size = train_params_dict.get('BATCH_SIZE', 128)
        self.num_epochs = train_params_dict.get('NUM_EPOCHS', 1)
        self.momentum = train_params_dict.get('MOMENTUM', 0.9)
        self.dropout_keep_prob = train_params_dict('DROPOUT_KEEP_PROB', 0.75)
        self.strategy = train_params_dict.get(
            'STRATEGY', tf.train.AdamOptimizer
        )

        self.model = model
        self.img_depth = img_params_dict.get('IMG_DEPTH', 2)
        self.views = ['x', 'u', 'v']

    # TODO - careful about separating target_label and the number of classes
    def run_training(
            self, target_label='segments', do_validation=False, short=False
    ):
        """
        run training (TRAIN file list) and optionally run a validation pass
        (on the VALID file list)
        """
        logger.info('staring run_training...')
        tf.reset_default_graph()
        initial_step = 0
        ckpt_dir = self.save_model_directory + '/checkpoints'
        run_dest_dir = self.save_model_directory + '/%d' % time.time()
        logger.info('tensorboard command:')
        logger.info('\ttensorboard --logdir {}'.format(
            self.save_model_directory
        ))

        with tf.Graph().as_default() as g:
            img_depth = self.img_depth
            train_reader = MnvDataReaderVertexST(
                filenames_list=self.train_file_list,
                batch_size=self.batch_size,
                name='train',
                compression=self.file_compression
            )
            batch_dict_train = train_reader.shuffle_batch_generator(
                num_epochs=self.num_epochs
            )
            X_train = batch_dict_train[self.features['x']]
            U_train = batch_dict_train[self.features['u']]
            V_train = batch_dict_train[self.features['v']]
            targ_train = batch_dict_train[self.targets_label]
            f_train = [X_train, U_train, V_train]

            valid_reader = MnvDataReaderVertexST(
                filenames_list=self.valid_file_list,
                batch_size=self.batch_size,
                name='valid',
                compression=self.file_compression
            )
            batch_dict_valid = valid_reader.batch_generator(num_epochs=1)
            X_valid = batch_dict_valid[self.features['x']]
            U_valid = batch_dict_valid[self.features['u']]
            V_valid = batch_dict_valid[self.features['v']]
            targ_valid = batch_dict_valid[self.targets_label]
            f_valid = [X_valid, U_valid, V_valid]

            d = make_default_convpooldict(img_depth=img_depth)
            model = TriColSTEpsilon(
                learning_rate=self.learning_rate,
                n_classes=11
            )
            model.prepare_for_inference(f_train, d)
            model.prepare_for_training(targ_train)

            writer = tf.summary.FileWriter(run_dest_dir)
            saver = tf.train.Saver()

            # n_steps: control this with num_epochs
            n_steps = 5 if short else 1e9
            skip_step = 1 if short else self.save_freq
            logger.info(' Processing {} steps...'.format(n_steps))

            init = tf.global_variables_initializer()

            with tf.Session(graph=g) as sess:
                start_time = time.time()
                sess.run(init)
                # have to run local variable init for `string_input_producer`
                sess.run(tf.local_variables_initializer())

                ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    logger.info('Restored session from {}'.format(ckpt_dir))

                writer.add_graph(sess.graph)
                initial_step = model.global_step.eval()
                logger.info('initial step =', initial_step)
                average_loss = 0.0

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                # NOTE: specifically catch `tf.errors.OutOfRangeError` or we
                # won't handle the exception correctly.
                try:
                    for b_num in range(initial_step, initial_step + n_steps):
                        _, loss_batch, summary = sess.run(
                            [model.optimizer,
                             model.loss,
                             model.train_summary_op],
                            feed_dict={
                                model.dropout_keep_prob:
                                self.dropout_keep_prob
                            }
                        )
                        writer.add_summary(summary, global_step=b_num)
                        average_loss += loss_batch
                        if (b_num + 1) % skip_step == 0:
                            logger.info(
                                '  Avg train loss at step {}: {:5.1f}'.format(
                                    b_num + 1, average_loss / skip_step
                                )
                            )
                            logger.info('   Elapsed time = {}'.format(
                                time.time() - start_time
                            ))
                            average_loss = 0.0
                            saver.save(sess, ckpt_dir, b_num)
                            logger.info('     saved at iter %d' % b_num)
                            # try validation
                            model.reassign_features(f_valid)
                            model.reassign_targets(targ_valid)
                            loss_valid, summary = sess.run(
                                [model.loss,
                                 model.valid_summary_op],
                                feed_dict={
                                    model.dropout_keep_prob: 1.0
                                }
                            )
                            writer.add_summary(summary, global_step=b_num)
                            logger.info('   Valid loss =', loss_valid)
                            # reset for training
                            model.reassign_features(f_train)
                            model.reassign_targets(targ_train)
                except tf.errors.OutOfRangeError:
                    logger.info('Training stopped - queue is empty.')
                except Exception as e:
                    logger.error(e)
                finally:
                    coord.request_stop()
                    coord.join(threads)

            writer.close()

        logger.info('Finished training...')

    def run_testing(self, short=False):
        """
        run a test pass (not "validation"!), based on the TEST file list.
        """
        logger.info('Starting testing...')
        tf.reset_default_graph()
        ckpt_dir = self.save_model_directory + '/checkpoints'

        with tf.Graph().as_default() as g:
            img_depth = self.img_depth
            data_reader = MnvDataReaderVertexST(
                filenames_list=self.test_file_list,
                batch_size=self.batch_size,
                name='test',
                compression=self.file_compression
            )
            batch_dict = data_reader.batch_generator()
            X = batch_dict[self.features['x']]
            U = batch_dict[self.features['u']]
            V = batch_dict[self.features['v']]
            targ = batch_dict[self.targets_label]
            f = [X, U, V]

            d = make_default_convpooldict(img_depth=img_depth)
            model = TriColSTEpsilon(
                learning_rate=self.learning_rate,
                n_classes=self.n_classes
            )
            model.prepare_for_inference(f, d)
            model.prepare_for_loss_computation(targ)

            saver = tf.train.Saver()

            n_batches = 2 if short else 10000
            init = tf.global_variables_initializer()
            logger.info(' Processing {} batches...'.format(n_batches))

            start_time = time.time()

            with tf.Session(graph=g) as sess:
                sess.run(init)
                # have to run local variable init for `string_input_producer`
                sess.run(tf.local_variables_initializer())

                ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    logger.info('Restored session from {}'.format(ckpt_dir))
                    if self.be_verbose:
                        logger.debug(
                            [op.name for op in
                             tf.get_default_graph().get_operations()]
                        )

                final_step = model.global_step.eval()
                logger.info('evaluation after {} steps.'.format(final_step))
                average_loss = 0.0
                total_correct_preds = 0

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                # NOTE: specifically catch `tf.errors.OutOfRangeError` or we
                # won't handle the exception correctly.
                n_processed = 0

                try:
                    for i in range(n_batches):
                        loss_batch, logits_batch, Y_batch = sess.run(
                            [model.loss, model.logits, targ],
                            feed_dict={
                                model.dropout_keep_prob: 1.0
                            }
                        )
                        n_processed += self.batch_size
                        average_loss += loss_batch
                        preds = tf.nn.softmax(logits_batch)
                        correct_preds = tf.equal(
                            tf.argmax(preds, 1), tf.argmax(Y_batch, 1)
                        )
                        if self.be_verbose:
                            logger.debug('   preds   = \n{}'.format(
                                tf.argmax(preds, 1).eval()
                            ))
                            logger.debug('   Y_batch = \n{}'.format(
                                tf.argmax(Y_batch, 1).eval()
                            ))
                        accuracy = tf.reduce_sum(
                            tf.cast(correct_preds, tf.float32)
                        )
                        total_correct_preds += sess.run(accuracy)
                        if self.be_verbose:
                            logger.debug(
                                '  batch {} loss = {} for nproc {}'.format(
                                    i, loss_batch, n_processed
                                )
                            )

                        logger.info(
                            "  total_correct_preds =", total_correct_preds
                        )
                        logger.info("  n_processed =", n_processed)
                        logger.info(" Accuracy {0}".format(
                            total_correct_preds / n_processed
                        ))
                        logger.info(' Average loss: {:5.1f}'.format(
                            average_loss / n_batches
                        ))
                except tf.errors.OutOfRangeError:
                    logger.info('Testing stopped - queue is empty.')
                except Exception as e:
                    logger.error(e)
                finally:
                    coord.request_stop()
                    coord.join(threads)

            logger.info('  Elapsed time = {}'.format(time.time() - start_time))

        logger.info('Finished testing...')

    def run_prediction(self):
        """
        make predictions based on the TEST file list
        """
        pass
