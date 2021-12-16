#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import os
import logging
import argparse
import ConfigParser
import common.datetime_wrapper as datetime
import common.hadoop_shell_wrapper as hadoop_shell
import common.fs_wrapper as fs
from model_configuration import ModelConfiguration
from dssm import DssmModel

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
        config.get(
            "model", "local_dir", 0,
            {
                "date": args.date,
                "project": config.get("common", "project")
            }
        )
    )

    local_model_path = os.path.join(local_train_path, "model")

    local_train_succ_file = os.path.join(local_train_path, "_TRAIN_SUCCESS")
    if fs.exists(local_train_succ_file):
        logging.info("succ flag found, skip training...")
        return 0

    # 清理model checkpoints目录
    if not fs.remove_dir(local_model_path) or not fs.mkdir(local_model_path):
        logging.error("fail to clean model dir %s", local_model_path)
        return 1

    hdfs_train_samples_path = config.get("common", "hdfs_output", 0, {
        "date": args.date,
        "dir": config.get("train_sample", "tfrecord_dir")
    })
    hdfs_test_samples_path = config.get("common", "hdfs_output", 0, {
        "date": args.date,
        "dir": config.get("test_sample", "tfrecord_dir")
    })

    # 检查训练样本是否存在集群上
    if not hadoop_shell.exists_all(
            hdfs_train_samples_path,
            flag=True,
            print_missing=True):
        logging.error("train samples missing on hdfs")
        return 1

    # 加载训练配置
    model_conf_file = os.path.join(local_train_path, "model.conf")
    if not fs.exists(model_conf_file):
        logging.error("model training conf file not found")
        return 1
    model_conf = ModelConfiguration().load(model_conf_file)
    model_conf.show()
    if not model_conf.check():
        logging.error("model conf invalid!")
        return 1

    if model_conf.test_fraction > 0 and not hadoop_shell.exists_all(
            hdfs_test_samples_path,
            flag=True,
            print_missing=True):
        logging.error("test samples missing on hdfs")
        return 1

    yarn_masters = eval(config.get("yarn_rec", "masters"))
    valid_master = hadoop_shell.first_exist(*yarn_masters, print_cmd=True, flag=False)
    if not valid_master:
        logging.error("no yarn master available")
        return 1
    modules_and_feeds = {"train": valid_master + hdfs_train_samples_path}
    if model_conf.test_fraction > 0:
        modules_and_feeds["test"] = valid_master + hdfs_test_samples_path

    # 训练模型
    model = DssmModel(model_conf, local_model_path, **modules_and_feeds)
    if model.train() != 0:
        logging.error("training failed")
        return 1

    logging.info("training succ")
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
