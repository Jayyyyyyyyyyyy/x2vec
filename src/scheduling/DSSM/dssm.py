#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf

import sys
sys.path.append("../")

import time
import os
import argparse
import common.fs_wrapper as fs
import numpy as np
import common.hadoop_shell_wrapper as hadoop
from model_configuration import ModelConfiguration
from sklearn.metrics import roc_auc_score
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DssmMetrics(object):

    @staticmethod
    def calc_auc(positive_sims, negative_sims):
        assert positive_sims.shape[0] == negative_sims.shape[0]
        aucs = []
        labels = [1] * positive_sims.shape[1] + [0] * negative_sims.shape[1]
        for i in xrange(positive_sims.shape[0]):
            scores = list(positive_sims[i]) + list(negative_sims[i])
            aucs.append(roc_auc_score(labels, scores))
        return np.mean(aucs)

    @staticmethod
    def calc_accuracy(positive_sims, negative_sims):
        assert positive_sims.shape[0] == negative_sims.shape[0]
        good = 0
        for i in xrange(positive_sims.shape[0]):
            if positive_sims[i][0] > max(negative_sims[i]):
                good += 1
        return good * 1.0 / len(positive_sims)


class StreamingMetric(object):
    def __init__(self, buffer_size):
        self._buffer_size = buffer_size
        self._streaming_metrics = []

    def add(self, val):
        if val > 0:
            self._streaming_metrics.append(val)
            if len(self._streaming_metrics) > self._buffer_size:
                self._streaming_metrics.pop(0)

    def get(self):
        return -1 if len(self._streaming_metrics) == 0 else sum(self._streaming_metrics) / len(self._streaming_metrics)


class DssmModel(object):
    def __init__(self, model_conf, model_dir, **modules_and_feeds):
        self.model_conf = model_conf
        self.model_dir = model_dir
        self.modules = modules_and_feeds

        fs.mkdir(model_dir)

        self.__init_variables()
        for module, sample_hdfs_path in modules_and_feeds.viewitems():
            self.__build_specified_graph(module, sample_hdfs_path)
        self.__init_or_load_session()

    def __init_or_load_session(self):
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))
        self.saver = tf.train.Saver(max_to_keep=self.model_conf.model_keep)
        checkpoint = tf.train.get_checkpoint_state(self.model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            tf.logging.warning("checkpoint loaded: %s", checkpoint.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

    def __parse_tfrecord_example(self, serialized_example):
        features = {
            "diu_list": tf.FixedLenFeature([], tf.string),
            "positives": tf.FixedLenFeature([self.model_conf.batch_size], tf.int64)
        }

        seq_features = {
            "play": self.model_conf.vocab_vid_size
        }
        for feature in seq_features.keys():
            ind_key = "{}_ind".format(feature)
            val_key = "{}_val".format(feature)
            features[ind_key] = tf.FixedLenFeature([], tf.string)
            features[val_key] = tf.FixedLenFeature([], tf.string)

        parsed_feats = tf.parse_single_example(serialized_example, features)

        out = {}
        for feature, vocab_size in seq_features.viewitems():
            ind_key = "{}_ind".format(feature)
            val_key = "{}_val".format(feature)
            inds = tf.decode_raw(parsed_feats[ind_key], tf.int32)
            inds = tf.cast(tf.reshape(inds, [-1, 2]), dtype=tf.int64)
            vals = tf.decode_raw(parsed_feats[val_key], tf.int32)
            # [batch_size, vocab_size]
            out[feature] = tf.compat.v1.SparseTensor(
                indices=inds,
                values=vals,
                dense_shape=tf.constant([self.model_conf.batch_size, vocab_size], dtype=tf.int64))
        out["diu_list"] = parsed_feats["diu_list"]
        out["positives"] = tf.reshape(parsed_feats["positives"], [self.model_conf.batch_size, 1])
        out["negatives"] = tf.concat(
            [
                tf.random_uniform(
                    (self.model_conf.batch_size, 1),
                    minval=0, maxval=200, dtype=tf.int64),
                tf.random_uniform(
                    (self.model_conf.batch_size, 1),
                    minval=0, maxval=1000, dtype=tf.int64),
                tf.random_uniform(
                    (self.model_conf.batch_size, 1),
                    minval=0, maxval=5000, dtype=tf.int64),
                tf.random_uniform(
                    (self.model_conf.batch_size, 1),
                    minval=0, maxval=50000, dtype=tf.int64),
                tf.random_uniform(
                    (self.model_conf.batch_size, self.model_conf.negative_samples - 4),
                    minval=0, maxval=self.model_conf.vocab_vid_size, dtype=tf.int64)
            ],
            axis=1
        )
        return out

    def __batch_graph_inputs(self, tfrecord_files, epoches=None):
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.shuffle(buffer_size=20) \
            .map(self.__parse_tfrecord_example) \
            .prefetch(buffer_size=20) \
            .repeat(epoches)
        return dataset.make_one_shot_iterator().get_next()

    def __init_variables(self):
        self.all_layers_sizes = []
        self.all_layers_sizes.append(self.model_conf.embed_vid_size)
        self.all_layers_sizes.extend(self.model_conf.fc_layers)
        self.all_layers_sizes.append(self.model_conf.embed_vid_size)
        self.fc_w_list = []
        self.fc_b_list = []
        self.embeddings = {}

        with tf.device('/gpu:{}'.format(self.model_conf.gpu_core)):
            with tf.variable_scope("fc_layers"):
                for layer_idx in xrange(1, len(self.all_layers_sizes)):
                    layer_w_name = "fc_W_{}".format(layer_idx)
                    self.fc_w_list.append(
                        tf.get_variable(
                            name=layer_w_name,
                            shape=[
                                self.all_layers_sizes[layer_idx - 1],
                                self.all_layers_sizes[layer_idx]
                            ],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32)
                    )
                    layer_b_name = "fc_b_{}".format(layer_idx)
                    self.fc_b_list.append(
                        tf.get_variable(
                            name=layer_b_name,
                            shape=[self.all_layers_sizes[layer_idx]],
                            initializer=tf.constant_initializer(0),
                            dtype=tf.float32)
                    )

            with tf.variable_scope("embeddings"):
                # define embeddings
                self.embeddings['vid'] = tf.get_variable(
                    "emb_vid",
                    shape=[self.model_conf.vocab_vid_size, self.model_conf.embed_vid_size],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    dtype=tf.float32)

    def __build_graph(self, sample_hdfs_path, epoches=None):
        tfrecords_files = hadoop.ls(sample_hdfs_path, path_regex=r".*/part-[0-9]{5}\.tfrecord")
        if not isinstance(tfrecords_files, list) or len(tfrecords_files) == 0:
            tf.logging.error("no input found!")
            return 1
        tfrecords_files = [x.name for x in tfrecords_files]

        inputs = self.__batch_graph_inputs(tfrecords_files, epoches=epoches)
        x_play_batch = inputs['play']
        x_diu_batch = inputs["diu_list"]
        # y_positives_batch -> [batch_size, 1]
        # y_negatives_batch -> [batch_size, negative_samples]
        y_positives_batch = inputs['positives']
        y_negatives_batch = inputs['negatives']

        with tf.device('/gpu:{}'.format(self.model_conf.gpu_core)):

            # feed multi one-hot inputs into embedding layer
            # [batch_size, embed_size]
            project_embedding_play_seq = \
                tf.nn.embedding_lookup_sparse(
                    self.embeddings['vid'],
                    sp_ids=x_play_batch,
                    sp_weights=None,
                    combiner="mean")

            # concat all embedding layers
            forward_layer = project_embedding_play_seq

            # pass through all fc layers
            for i in xrange(len(self.fc_w_list)):
                # forward_layer -> [batch_size, embed_vid_size]
                forward_layer = tf.nn.tanh(tf.matmul(forward_layer, self.fc_w_list[i]) + self.fc_b_list[i])

            # [batch_size, 1, embed_vid_size]
            positives_embeddings = tf.nn.embedding_lookup(self.embeddings["vid"], y_positives_batch)
            # [batch_size, negative_samples, embed_vid_size]
            negatives_embeddings = tf.nn.embedding_lookup(self.embeddings["vid"], y_negatives_batch)

            # [batch_size, 1]
            positive_sims = DssmModel.__cosine(
                tf.expand_dims(forward_layer, 1),
                positives_embeddings
            )
            # [batch_size, negative_samples]
            negative_sims = DssmModel.__cosine(
                tf.tile(
                    tf.expand_dims(forward_layer, 1),  # [batch_size, 1, embd_size]
                    [1, self.model_conf.negative_samples, 1]
                ),  # [batch_size, negative_samples, embd_size]
                negatives_embeddings
            )
            # [batch_size, negative_samples]
            aligned_positive_sims = tf.tile(positive_sims, [1, self.model_conf.negative_samples])

            # triplet loss
            loss = tf.reduce_mean(
                tf.maximum(
                    0.0,
                    self.model_conf.triplet_margin + negative_sims - aligned_positive_sims
                )
            )

            l2_reg_rate = self.model_conf.l2_reg_rate
            if l2_reg_rate > 0:
                regularizer = tf.contrib.layers.l2_regularizer(l2_reg_rate)
                for each_fc_w in self.fc_w_list:
                    loss = loss + regularizer(each_fc_w)

            loss = loss
            user_embedding = DssmModel.__normalize(forward_layer)
            video_embedding = DssmModel.__normalize(self.embeddings["vid"])
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.model_conf.learning_rate
            ).minimize(loss)
            return optimizer, loss, positive_sims, negative_sims, x_diu_batch, user_embedding, video_embedding

    def __build_specified_graph(self, module, sample_hdfs_path):
        if module == "train":
            self.train_optimizer, self.train_loss, self.train_positive_sims, self.train_negative_sims, _, _, _ = \
                self.__build_graph(sample_hdfs_path, epoches=self.model_conf.epoches)
        elif module == "test":
            _, self.test_loss, self.test_positive_sims, self.test_negative_sims, _, _, _ = \
                self.__build_graph(sample_hdfs_path, epoches=None)
        elif module == "inference":
            _, _, _, _, self.x_diu_batch, self.user_embedding_op, self.video_embedding_op = \
                self.__build_graph(sample_hdfs_path, epoches=1)
        else:
            raise Exception("unknown module {}/{}".format(module, sample_hdfs_path))

    @staticmethod
    def __normalize(matrix):
        return matrix / tf.reshape(
            tf.sqrt(tf.reduce_sum(tf.square(matrix), axis=1) + 0.0001),
            [tf.shape(matrix)[0], 1]
        )

    @staticmethod
    def __embedding_avg_pooling(x, embedding_matrix, input_size, embed_size):
        """
        :param x: [batch_size, input_size]
        :param embedding_matrix: [vocab_size, embed_size]
        :param input_size: int
        :param embed_size: int
        :return: [batch_size, embed_size]
        """
        embed = tf.nn.embedding_lookup(embedding_matrix, x)
        mask = tf.tile(
            tf.reshape(tf.cast(tf.sign(x), tf.float32), [tf.shape(x)[0], input_size, -1]),
            [1, 1, embed_size])
        masked_embed = tf.multiply(embed, mask)
        return tf.div(tf.reduce_sum(masked_embed, axis=1),  tf.reduce_sum(mask, axis=1))

    @staticmethod
    def __cosine(tensor_mat1, tensor_mat2):
        """
        cosine similarity bewteen tow tensors
        :param tensor_mat1: [batch_size, None, embed_size]
        :param tensor_mat2: [batch_size, None, embed_size]
        :return: [batch_size, None]
        """
        # [batch_size, None]
        s1 = tf.sqrt(tf.reduce_sum(tf.square(tensor_mat1), axis=2) + 0.0001)
        # [batch_size, None]
        s2 = tf.sqrt(tf.reduce_sum(tf.square(tensor_mat2), axis=2) + 0.0001)
        # [batch_size, None]
        base = tf.reduce_sum(tf.multiply(tensor_mat1, tensor_mat2), axis=2)
        cos = tf.divide(base, tf.multiply(s1, s2))
        return cos

    def __train_batch(self, should_test):
        auc = -1
        accuracy = -1
        if should_test:
            _, batch_loss, batch_positive_sims, batch_negative_sims = \
                self.sess.run([
                    self.train_optimizer,
                    self.train_loss,
                    self.train_positive_sims,
                    self.train_negative_sims])
            auc = DssmMetrics.calc_auc(batch_positive_sims, batch_negative_sims)
            accuracy = DssmMetrics.calc_accuracy(batch_positive_sims, batch_negative_sims)
        else:
            _, batch_loss = self.sess.run([self.train_optimizer, self.train_loss])
        return batch_loss, auc, accuracy

    def __test_batch(self):
        batch_loss, batch_positive_sims, batch_negative_sims = \
            self.sess.run([self.test_loss, self.test_positive_sims, self.test_negative_sims])
        auc = DssmMetrics.calc_auc(batch_positive_sims, batch_negative_sims)
        accuracy = DssmMetrics.calc_accuracy(batch_positive_sims, batch_negative_sims)
        return batch_loss, auc, accuracy

    def train(self):
        assert "train" in self.modules
        assert self.sess is not None
        assert self.train_optimizer is not None and \
            self.train_loss is not None and \
            self.train_positive_sims is not None and \
            self.train_negative_sims is not None

        total_batch = 0
        start_ts = int(round(time.time() * 1000))

        try:
            train_loss_stream = StreamingMetric(50)
            train_auc_stream = StreamingMetric(20)
            train_accuracy_stream = StreamingMetric(20)
            test_loss_stream = StreamingMetric(50)
            test_auc_stream = StreamingMetric(20)
            test_accuracy_stream = StreamingMetric(20)

            while True:
                should_test = self.model_conf.eval_step <= 0 or (total_batch + 1) % self.model_conf.eval_step == 0

                train_batch_loss, train_auc, train_accuracy = self.__train_batch(should_test)
                test_batch_loss, test_auc, test_accuracy = self.__test_batch() \
                    if "test" in self.modules and should_test else (-1, -1, -1)

                total_batch += 1
                train_loss_stream.add(train_batch_loss)
                train_auc_stream.add(train_auc)
                train_accuracy_stream.add(train_accuracy)
                test_loss_stream.add(test_batch_loss)
                test_auc_stream.add(test_auc)
                test_accuracy_stream.add(test_accuracy)

                if should_test:
                    cur_ts = int(round(time.time() * 1000))
                    rps = total_batch * self.model_conf.batch_size * 1000.0 / (cur_ts - start_ts)
                    tf.logging.warning(
                        "train batch %d: speed = %f rps, passed = %d, "
                        "loss = (%.3f | %.3f), auc = (%.3f | %.3f), accuracy = (%.3f | %.3f)",
                        total_batch, rps, total_batch * self.model_conf.batch_size,
                        train_loss_stream.get(), test_loss_stream.get(),
                        train_auc_stream.get(), test_auc_stream.get(),
                        train_accuracy_stream.get(), test_accuracy_stream.get())

                # dump model
                if total_batch % self.model_conf.checkpoint_steps == 0:
                    self.saver.save(self.sess, self.model_dir + '/model.ckpt', global_step=total_batch)

        except tf.errors.OutOfRangeError:
            tf.logging.warning("training finished!")

        if total_batch % self.model_conf.checkpoint_steps != 0:
            self.saver.save(self.sess, self.model_dir + '/model.ckpt', global_step=total_batch)

        return 0 if total_batch > 0 else 1

    def export_embedding(self, output_filename):
        tf.logging.warning("===== start to persist embedding =====")
        assert self.sess is not None and "inference" in self.modules
        embedding_matrix = self.sess.run(self.video_embedding_op)
        with open(output_filename, "w") as file_out:
            idx = 0
            for emb in embedding_matrix:
                file_out.write("{}\t{}\n".format(
                    idx,
                    ",".join(map(lambda x: "{:.6f}".format(x), emb))
                ))
                idx += 1
        tf.logging.warning("===== finished exporting embedding =====")
        return 0

    def inference(self, output_filename):
        assert self.sess is not None and "inference" in self.modules

        total_batch = 0
        start_ts = int(round(time.time() * 1000))

        with open(output_filename, "w") as fout:
            try:
                while True:
                    diu_str, user_emb = self.sess.run([self.x_diu_batch, self.user_embedding_op])

                    diu_list = diu_str.split("\x01")
                    if len(diu_list) != len(user_emb):
                        raise Exception(
                            "diu list and user emb size mismatch in one batch, [diu={}], [emb={}]".format(
                                len(diu_list),
                                len(user_emb)
                            )
                        )
                    num = len(diu_list)
                    for i in xrange(num):
                        fout.write(
                            "{}\t{}\n".format(
                                diu_list[i],
                                ",".join(map(lambda x: "{:.6f}".format(x), user_emb[i]))
                            )
                        )

                    total_batch += 1
                    cur_ts = int(round(time.time() * 1000))
                    rps = total_batch * self.model_conf.batch_size * 1000.0 / (cur_ts - start_ts)
                    if total_batch % 200 == 0:
                        tf.logging.warning(
                            "inference batch %d: speed = %f rps, passed = %d",
                            total_batch,
                            rps,
                            total_batch * self.model_conf.batch_size)

            except tf.errors.OutOfRangeError:
                tf.logging.warning("inference finished!")
            except Exception as e:
                tf.logging.error(e)
                return 1
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument("--model", dest='model', required=True, help="local model dir")
    parser.add_argument("--train_sample", dest='train_sample', required=True, help="sample hdfs absolute path")
    parser.add_argument("--test_sample", dest='test_sample', required=True, help="sample hdfs absolute path")
    arguments = parser.parse_args()

    try:
        model_conf = ModelConfiguration().load(arguments.model + "/model.conf")
        if not model_conf.check():
            raise Exception("invalid model info")
        model_conf.show()

        modules_and_feeds = {"train": arguments.train_sample, "test": arguments.test_sample}
        dssm = DssmModel(model_conf, arguments.model + "/model", **modules_and_feeds)
        sys.exit(dssm.train())
    except Exception as ex:
        tf.logging.error(ex)
        sys.exit(1)
