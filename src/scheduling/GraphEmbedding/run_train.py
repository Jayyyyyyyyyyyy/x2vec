#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import os
import json
import logging
import argparse
import ConfigParser
import common.datetime_wrapper as datetime
import common.hadoop_shell_wrapper as hadoop_shell
import common.fs_wrapper as fs
import common.shell_wrapper as shell
from gensim.models import word2vec, Word2Vec

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def run(args):

    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    local_train_path = "{}/../{}".format(
        FILE_DIR,
        config.get("model", "local_dir", 0,
                   {
                       "date": args.date,
                       "project": config.get("common", "project")
                   }))

    local_train_sample_file = "{}/{}".format(local_train_path, "train.sample")
    local_model_path = os.path.join(local_train_path, "model")
    local_model_file = os.path.join(local_model_path, "word2vec.model")
    local_train_succ_file = "{}/{}".format(local_train_path, "_TRAIN_SUCCESS")
    hdfs_train_samples_path = config.get("common", "hdfs_output", 0, {
        "date": args.date, "dir": config.get("deep_walk", "dir")
    })

    if fs.exists(local_train_succ_file):
        logging.info("train succ file found, skip training")
        return 0
    # 清理本地训练目录
    if not fs.remove_dir(local_train_path) or not fs.mkdir(local_train_path):
        logging.error("fail to clean training dir %s", local_train_path)
        return 1
    if not fs.mkdir(local_model_path):
        logging.error("fail to clean word2vec model dir %s", local_model_path)
        return 1

    # 下载训练样本
    if not hadoop_shell.exists_all(hdfs_train_samples_path, flag=True, print_missing=True):
        logging.error("train sample not found on hdfs")
        return 1

    if not hadoop_shell.getmerge(hdfs_train_samples_path, local_train_sample_file, print_cmd=True):
        logging.error("fail to download train sample from %s", hdfs_train_samples_path)
        return 1

    # 训练word2vec
    sequences = word2vec.Text8Corpus(local_train_sample_file)
    # 无需设置min_count来过滤低频词，已经在deep_walk阶段处理
    model = word2vec.Word2Vec(
        sequences,
        size=config.getint("word2vec", "size"),  # embedding dim
        window=config.getint("word2vec", "window"),  # 上下文窗口半径
        alpha=config.getfloat("word2vec", "alpha"),  # 初始学习率
        min_alpha=config.getfloat("word2vec", "min_alpha"),  # 最小学习率
        sg=config.getint("word2vec", "sg"),  # skip-gram is employed (0 for CBOW)
        hs=config.getint("word2vec", "hs"),  # negative sampling (1 for hierarchical softmax)
        negative=config.getint("word2vec", "negative"),  # how many “noise words” should be drawn
        workers=config.getint("word2vec", "workers"),  # worker threads
        iter=config.getint("word2vec", "iter"))  # number of iterations (epochs) over the corpus
    model.save(local_model_file)
    fs.touch(local_train_succ_file)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train bpr embedding')
    parser.add_argument('--date', dest='date', required=True, help="which date(%%Y-%%m-%%d)")
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    arguments = parser.parse_args()

    try:
        if not datetime.DateTime(arguments.date).is_perfect_date():
            raise RuntimeError("passed arg [date={}] format error".format(arguments.date))
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
