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
import common.shell_wrapper as shell

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def model_id_mapping(input_model_file, input_id_mapping, output_model_file):
    mapping = dict()
    with open(input_id_mapping, "r") as fin:
        for line in fin:
            items = line.strip().split("\t", 1)
            mapping[items[1]] = items[0]

    total = 0
    hit = 0
    with open(output_model_file, "w") as fout:
        with open(input_model_file, "r") as fin:
            for line in fin:
                items = line.strip().split(" ")
                total += 1
                if len(items) > 1 and items[0] in mapping:
                    print >> fout, "{}\t{}".format(mapping[items[0]], ",".join(items[1:]))
                    hit += 1

    hit_rate = hit * 1.0 / total
    logging.info("id mapping on %s, hit rate = %f", input_model_file, hit_rate)
    return hit_rate > 0.99


def run(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    local_train_path = "{}/../{}".format(
        FILE_DIR,
        config.get(
            "training", "local_dir", 0,
            {"date": args.date, "project": config.get("common", "project")}))

    local_train_samples_file = "{}/{}".format(local_train_path, "train.samples")
    local_test_samples_file = "{}/{}".format(local_train_path, "test.samples")
    local_user_model_file = "{}/{}".format(local_train_path, "user.emb")
    local_item_model_file = "{}/{}".format(local_train_path, "item.emb")
    local_user_model_file_raw = local_user_model_file + ".raw"
    local_item_model_file_raw = local_item_model_file + ".raw"
    local_user_ids_file = "{}/{}".format(local_train_path, "user.ids")
    local_item_ids_file = "{}/{}".format(local_train_path, "item.ids")

    local_train_succ_file = "{}/{}".format(local_train_path, "_TRAIN_SUCCESS")

    if fs.exists(local_train_succ_file):
        logging.info("succ flag found, skip training...")
        return 0

    if not fs.remove_dir(local_train_path) or not fs.mkdir(local_train_path):
        logging.error("fail to clean training dir %s", local_train_path)
        return 1

    hdfs_train_samples_path = config.get("common", "hdfs_output", 0,
                                         {'date': args.date, 'dir': config.get("train_samples", "dir")})
    hdfs_test_samples_path = config.get("common", "hdfs_output", 0,
                                        {'date': args.date, 'dir': config.get("test_samples", "dir")})
    hdfs_user_ids_path = config.get("common", "hdfs_output", 0,
                                    {'date': args.date, 'dir': config.get("user_ids", "dir")})
    hdfs_item_ids_path = config.get("common", "hdfs_output", 0,
                                    {'date': args.date, 'dir': config.get("item_ids", "dir")})

    # 下载训练/测试数据
    if not hadoop_shell.exists_all(hdfs_train_samples_path,
                                   hdfs_test_samples_path,
                                   hdfs_user_ids_path,
                                   hdfs_item_ids_path,
                                   flag=True, print_missing=True):
        logging.error("train/test samples missing on hdfs")
        return 1

    if not hadoop_shell.getmerge(hdfs_train_samples_path, local_train_samples_file, print_cmd=True):
        logging.error("fail to download train sample from %s", hdfs_train_samples_path)
        return 1
    if not hadoop_shell.getmerge(hdfs_test_samples_path, local_test_samples_file, print_cmd=True):
        logging.error("fail to download test sample from %s", hdfs_test_samples_path)
        return 1
    if not hadoop_shell.getmerge(hdfs_user_ids_path, local_user_ids_file, print_cmd=True):
        logging.error("fail to download user ids from %s", hdfs_user_ids_path)
        return 1
    if not hadoop_shell.getmerge(hdfs_item_ids_path, local_item_ids_file, print_cmd=True):
        logging.error("fail to download item ids from %s", hdfs_item_ids_path)
        return 1

    # 训练
    cmd = "{}/../bin/bpr --test_always --test_avg_metrics=auc " \
          "--train_dataset={} " \
          "--test_dataset={} " \
          "--user_factors={} " \
          "--item_factors={} " \
          "--nepochs={} " \
          "--nfactors={} " \
          "--nthreads={} " \
          "--num_hogwild_threads={} " \
          "--init_learning_rate={} " \
          "--decay_rate={} " \
          "--num_test_users={} " \
          "--user_lambda={} " \
          "--item_lambda={}".format(FILE_DIR,
                                    local_train_samples_file,
                                    local_test_samples_file,
                                    local_user_model_file_raw,
                                    local_item_model_file_raw,
                                    config.getint("bpr", "nepochs"),
                                    config.getint("bpr", "nfactors"),
                                    config.getint("bpr", "nthreads"),
                                    config.getint("bpr", "num_hogwild_threads"),
                                    config.getfloat("bpr", "init_learning_rate"),
                                    config.getfloat("bpr", "decay_rate"),
                                    config.getint("bpr", "num_test_users"),
                                    config.getfloat("bpr", "user_lambda"),
                                    config.getfloat("bpr", "item_lambda"))

    logging.info("training cmd: %s", cmd)
    if not shell.shell_command(cmd=cmd, print_info=True):
        logging.error("training [%s] failed", args.date)
        return 1
    logging.info("training [%s] succeeded, start id mapping", args.date)

    if not model_id_mapping(local_user_model_file_raw, local_user_ids_file, local_user_model_file) or \
            not model_id_mapping(local_item_model_file_raw, local_item_ids_file, local_item_model_file):
        logging.error("fail to perform id mapping on emb models")
        return 1
    logging.info("successfully id mapping")
    fs.remove_file(local_user_model_file_raw, local_item_model_file_raw,
                   local_train_samples_file, local_test_samples_file, local_user_ids_file, local_item_ids_file)

    fs.touch(local_train_succ_file)

    # 删除过期
    if args.expire and config.getint("common", "expire") > 0:
        lifetime = config.getint("training", "local_lifetime")
        if lifetime > 0:
            dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
            expire_path = "{}/../{}".format(
                FILE_DIR,
                config.get(
                    "training", "local_dir", 0,
                    {"date": dt_expire, "project": config.get("common", "project")}))
            if fs.remove_dir(expire_path):
                logging.info("successfully removed local expire %s", expire_path)
            else:
                logging.warning("failed to remove local expire %s", expire_path)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train bpr embedding')
    parser.add_argument('--date', dest='date', required=True, help="which date(%%Y-%%m-%%d)")
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    parser.add_argument('--expire', dest='expire', action="store_true", help="whether to del expired path")
    arguments = parser.parse_args()

    try:
        if not datetime.DateTime(arguments.date).is_perfect_date():
            raise RuntimeError("passed arg [date={}] format error".format(arguments.date))
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
