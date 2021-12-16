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
import common.spark_submit_wrapper as spark


FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def process_build_samples(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    ndays = config.getint("training", "n_days")
    assert ndays >= 1, "ndays must be >= 1"
    ts_bound = datetime.DateTime.to_ts(datetime.DateTime(args.date).apply_offset_by_day(-ndays))
    nlastest = config.getint("training", "n_lastest")

    item_freq_thres = config.getint("training", "item_freq_thres")
    user_confidence = config.getint("training", "user_confidence")
    test_fraction = -1 if args.ignore_test else config.getfloat("training", "test_fraction")

    input_path_user_history = config.get("user_history", "path", 0, {'date': args.date})
    output_train_samples = config.get("common", "hdfs_output", 0,
                                      {'date': args.date, 'dir': config.get("train_samples", "dir")})
    output_test_samples = config.get("common", "hdfs_output", 0,
                                      {'date': args.date, 'dir': config.get("test_samples", "dir")})
    output_user_ids = config.get("common", "hdfs_output", 0,
                                 {'date': args.date, 'dir': config.get("user_ids", "dir")})
    output_item_ids = config.get("common", "hdfs_output", 0,
                                 {'date': args.date, 'dir': config.get("item_ids", "dir")})

    if args.ignore_test and hadoop_shell.exists_all(output_train_samples, output_user_ids, output_item_ids, flag=True):
        logging.info("output already exists, skip")
        return 0

    if not args.ignore_test and hadoop_shell.exists_all(
            output_test_samples, output_train_samples, output_user_ids, output_item_ids, flag=True):
        logging.info("output already exists, skip")
        return 0

    if not hadoop_shell.exists_all_with_retry(input_path_user_history,
                                              flag=True,
                                              print_info=True,
                                              retry=config.getint("user_history", "retry"),
                                              interval=config.getint("user_history", "interval")):
        logging.error("finally, input not ready, exit!")
        return 1

    hadoop_shell.rmr(output_train_samples, output_test_samples, output_user_ids, output_item_ids)

    ss = spark.SparkSubmitWrapper()

    ss.set_master("yarn")\
        .set_deploy_mode("cluster")\
        .set_driver_memory("2G")\
        .set_executor_memory("2G") \
        .add_conf("spark.executor.memoryOverhead", 1024) \
        .set_executor_cores(2)\
        .set_num_executors(100)\
        .add_conf("spark.network.timeout", 600)\
        .set_name(config.get("common", "job_name", 0, {'date': args.date, 'module': "BuildSamples"}))\
        .set_queue(config.get("common", "job_queue"))\
        .set_class("com.td.ml.x2vec.bpr.BuildSample")\
        .set_app_jar(FILE_DIR + "/../lib/" + config.get("common", "jar"))\
        .add_app_argument("ts_bound", ts_bound)\
        .add_app_argument("nlastest", nlastest)\
        .add_app_argument("item_freq_thres", item_freq_thres) \
        .add_app_argument("test_fraction", test_fraction) \
        .add_app_argument("user_confidence", user_confidence)\
        .add_app_argument("input_path_user_history", input_path_user_history)\
        .add_app_argument("output_path_train_samples", output_train_samples)\
        .add_app_argument("output_path_test_samples", output_test_samples)\
        .add_app_argument("output_path_user_ids", output_user_ids)\
        .add_app_argument("output_path_item_ids", output_item_ids)

    return 0 if ss.run(print_cmd=True, print_info=True) else 1


def del_expire(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    if not args.expire or config.getint("common", "expire") <= 0:
        return 0

    for option in ["train_samples", "test_samples", "user_ids", "item_ids"]:
        lifetime = config.getint(option, "lifetime")
        if lifetime > 0:
            dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
            path_expire = config.get("common", "hdfs_output", 0,
                                     {'date': dt_expire, 'dir': config.get(option, "dir")})
            if not hadoop_shell.rmr(path_expire):
                logging.error("fail to del expired path: %s", path_expire)
                return 1
    return 0


def run(args):
    if process_build_samples(args) == 0 and del_expire(args) == 0:
        return 0
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build samples')
    parser.add_argument('--date', dest='date', required=True, help="which date(%%Y-%%m-%%d)")
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    parser.add_argument('--expire', dest='expire', action="store_true", help="whether to del expired path")
    parser.add_argument("--ignore_test", dest="ignore_test", action="store_true", help="will not build test samples")

    arguments = parser.parse_args()

    try:
        if not datetime.DateTime(arguments.date).is_perfect_date():
            raise RuntimeError("passed arg [date={}] format error".format(arguments.date))
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
