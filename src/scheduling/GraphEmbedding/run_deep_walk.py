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


def process_deep_walk(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    ndays = config.getint("deep_walk", "n_days")
    assert ndays >= 1, "ndays must be >= 1"
    ts_bound = datetime.DateTime.to_ts(datetime.DateTime(args.date).apply_offset_by_day(-ndays))
    seq_window_size = config.getint("deep_walk", "seq_window_size")
    vertex_freq_thres = config.getint("deep_walk", "vertex_freq_thres")
    edge_freq_thres = config.getint("deep_walk", "edge_freq_thres")
    seq_distinct_items = config.getint("deep_walk", "seq_distinct_items")
    adjacency_max_size = config.getint("deep_walk", "adjacency_max_size")
    walk_length = config.getint("deep_walk", "walk_length")
    walk_min_length = config.getint("deep_walk", "walk_min_length")
    walks_per_vertex = config.getint("deep_walk", "walks_per_vertex")

    input_path_user_history = config.get("user_history", "path", 0, {'date': args.date})
    input_path_video_profile = config.get("video_profile", "path", 0, {'date': args.date})

    output_path_deep_walk = config.get("common", "hdfs_output", 0, {
        "date": args.date,
        "dir": config.get("deep_walk", "dir")
    })
    output_path_vocab = config.get("common", "hdfs_output", 0, {
        "date": args.date,
        "dir": config.get("vocab", "dir")
    })
    output_path_adjacency_matrix = config.get("common", "hdfs_output", 0, {
        "date": args.date,
        "dir": config.get("adjacency_matrix", "dir")
    })

    if walks_per_vertex > 0 and hadoop_shell.exists_all(
            output_path_deep_walk,
            output_path_vocab,
            output_path_adjacency_matrix,
            flag=True):
        logging.info("output already exists, skip")
        return 0

    if walks_per_vertex <= 0 and hadoop_shell.exists_all(
            output_path_deep_walk,
            output_path_vocab,
            flag=True):
        logging.info("output already exists, skip")
        return 0

    if not hadoop_shell.exists_all_with_retry(input_path_user_history,
                                              input_path_video_profile,
                                              flag=True,
                                              print_info=True,
                                              retry=config.getint("common", "upstream_retry"),
                                              interval=config.getint("common", "upstream_interval")):
        logging.error("finally, input not ready, exit!")
        return 1

    if not hadoop_shell.rmr(output_path_vocab, output_path_deep_walk, output_path_adjacency_matrix):
        logging.error("fail to clear output path")
        return 1

    ss = spark.SparkSubmitWrapper()

    ss.set_master("yarn")\
        .set_deploy_mode("cluster")\
        .set_driver_memory("1G")\
        .set_executor_memory("2G") \
        .add_conf("spark.executor.memoryOverhead", 2048) \
        .set_executor_cores(2)\
        .set_num_executors(100)\
        .add_conf("spark.network.timeout", 600)\
        .set_name(config.get("common", "job_name", 0, {'date': args.date, 'module': "WeightedDeepWalk"}))\
        .set_queue(config.get("common", "job_queue"))\
        .set_class("com.td.ml.x2vec.GraphEmbedding.DeepWalkWeightedOnlyVideo")\
        .set_app_jar(FILE_DIR + "/../lib/" + config.get("common", "jar"))\
        .add_app_argument("ts_bound", ts_bound)\
        .add_app_argument("seq_window_size", seq_window_size)\
        .add_app_argument("edge_freq_thres", edge_freq_thres) \
        .add_app_argument("vertex_freq_thres", vertex_freq_thres) \
        .add_app_argument("adjacency_max_size", adjacency_max_size) \
        .add_app_argument("walk_length", walk_length)\
        .add_app_argument("walk_min_length", walk_min_length) \
        .add_app_argument("seq_distinct_items", seq_distinct_items) \
        .add_app_argument("walks_per_vertex", walks_per_vertex)\
        .add_app_argument("input_path_user_history", input_path_user_history)\
        .add_app_argument("input_path_video_profile", input_path_video_profile)\
        .add_app_argument("output_path_vocab", output_path_vocab)\
        .add_app_argument("output_path_deep_walk", output_path_deep_walk)\
        .add_app_argument("output_path_adjacency_matrix", output_path_adjacency_matrix)

    return 0 if ss.run(print_cmd=True, print_info=True) else 1


def del_expire(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    if not args.expire or config.getint("common", "expire") <= 0:
        return 0

    lifetime = config.getint("deep_walk", "lifetime")
    if lifetime > 0:
        dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
        expire_path = config.get("common", "hdfs_output", 0, {
            "date": dt_expire,
            "dir": config.get("deep_walk", "dir")
        })
        if not hadoop_shell.rmr(expire_path):
            logging.error("fail to del expired path: %s", expire_path)
            return 1

    lifetime = config.getint("vocab", "lifetime")
    if lifetime > 0:
        dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
        expire_path = config.get("common", "hdfs_output", 0, {
            "date": dt_expire,
            "dir": config.get("vocab", "dir")
        })
        if not hadoop_shell.rmr(expire_path):
            logging.error("fail to del expired path: %s", expire_path)
            return 1

    return 0


def run(args):
    if process_deep_walk(args) == 0 and del_expire(args) == 0:
        return 0
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run deep walk')
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
