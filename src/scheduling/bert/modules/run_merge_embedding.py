#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../../")
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


def process(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    yesterday = datetime.DateTime(args.date).apply_offset_by_day(-1)

    input_path_embedding_history = config.get("common", "hdfs_output", 0, {
        "date": yesterday,
        "dir": config.get("embedding_history", "dir")
    })
    input_path_incoming_embedding = config.get("inputs", "bert_embedding_kafka", 0, {"date": args.date})
    output_path = config.get("common", "hdfs_output", 0, {
        "date": args.date,
        "dir": config.get("embedding_history", "dir")
    })

    if hadoop_shell.exists_all(output_path, flag=True, print_missing=True):
        logging.info("output already exists, skip")
        return 0

    if not args.init and not hadoop_shell.exists_all(
            input_path_embedding_history,
            flag=True,
            print_missing=True,
            print_cmd=True):
        logging.error("finally, input not ready, exit!")
        return 1

    if not hadoop_shell.rmr(output_path):
        logging.error("fail to clear output folder")
        return 1

    ss = spark.SparkSubmitWrapper()

    ss.set_master("yarn")\
        .set_deploy_mode("cluster")\
        .set_driver_memory("1G")\
        .set_executor_memory("1G") \
        .add_conf("spark.executor.memoryOverhead", 2048) \
        .set_executor_cores(2)\
        .set_num_executors(100)\
        .add_conf("spark.network.timeout", 600)\
        .set_name(config.get("common", "job_name", 0, {'date': args.date, 'module': "MergeEmbedding"}))\
        .set_queue(config.get("common", "job_queue"))\
        .set_class("com.td.ml.x2vec.bert.MergeEmbedding")\
        .set_app_jar(FILE_DIR + "/../../lib/" + config.get("common", "jar"))\
        .add_app_argument("input_path_incoming_embedding", input_path_incoming_embedding)\
        .add_app_argument("output_path", output_path)

    if not args.init:
        ss.add_app_argument("input_path_embedding_history", input_path_embedding_history)

    return 0 if ss.run(print_cmd=True, print_info=True) else 1


def del_expire(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    if not args.expire or config.getint("common", "expire") <= 0:
        return 0

    lifetime = config.getint("embedding_history", "lifetime")

    if lifetime > 0:
        dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
        expire_path = config.get("common", "hdfs_output", 0, {
            "date": dt_expire,
            "dir": config.get("embedding_history", "dir")
        })
        if not hadoop_shell.rmr(expire_path):
            logging.error("fail to del expired path: %s", expire_path)
            return 1
    return 0


def run(args):
    if process(args) == 0 and del_expire(args) == 0:
        return 0
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build user inference samples')
    parser.add_argument('--date', dest='date', required=True, help="which date(%%Y-%%m-%%d)")
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    parser.add_argument('--expire', dest='expire', action="store_true", help="whether to del expired path")
    parser.add_argument('--init', dest='init', action="store_true")

    arguments = parser.parse_args()

    try:
        if not datetime.DateTime(arguments.date).is_perfect_date():
            raise RuntimeError("passed arg [date={}] format error".format(arguments.date))
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
