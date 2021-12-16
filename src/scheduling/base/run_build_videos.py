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


def process(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    module_option = "{}_videos".format(args.module)

    ndays = config.getint(module_option, "n_days")
    assert ndays >= 1, "ndays must be >= 1"
    require_recommendable = config.get(module_option, "require_recommendable")
    feed_only = config.get(module_option, "feed_only")
    ts_bound = datetime.DateTime.to_ts(datetime.DateTime(args.date).apply_offset_by_day(-ndays))
    click_threshold = config.getint(module_option, "click_threshold")
    star_level_threshold = config.getint(module_option, "star_level_threshold")
    big_video_only = config.get(module_option, "big_video_only")

    input_path_user_history = config.get("user_history", "path", 0, {'date': args.date})
    input_path_video_profile = config.get("video_profile","path", 0, {'date': args.date})
    output_path = config.get("common", "hdfs_output", 0, {
        "date": args.date,
        "dir": config.get(module_option, "dir")
    })

    if hadoop_shell.exists_all(output_path, flag=True, print_missing=True):
        logging.info("output already exists, skip")
        return 0

    if not hadoop_shell.exists_all_with_retry(
            input_path_user_history,
            flag=True,
            print_info=True,
            retry=config.getint("common", "upstream_retry"),
            interval=config.getint("common", "upstream_interval")):
        logging.error("finally, input not ready, exit!")
        return 1

    if require_recommendable and not hadoop_shell.exists_all_with_retry(
            input_path_video_profile, flag=True, print_info=True,
            retry=config.getint("common", "upstream_retry"),
            interval=config.getint("common", "upstream_interval")):
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
        .set_name(config.get("common", "job_name", 0, {'date': args.date, 'module': "BuildVideos-{}".format(args.module)}))\
        .set_queue(config.get("common", "job_queue"))\
        .set_class("com.td.ml.x2vec.base.CountVideo")\
        .set_app_jar(FILE_DIR + "/../lib/" + config.get("common", "jar"))\
        .add_app_argument("ts_bound", ts_bound) \
        .add_app_argument("click_threshold", click_threshold) \
        .add_app_argument("require_recommendable", require_recommendable)\
        .add_app_argument("feed_only", feed_only) \
        .add_app_argument("star_level_threshold", star_level_threshold)\
        .add_app_argument("big_video_only", big_video_only)\
        .add_app_argument("input_path_user_history", input_path_user_history)\
        .add_app_argument("input_path_video_profile", input_path_video_profile)\
        .add_app_argument("output_path", output_path)

    return 0 if ss.run(print_cmd=True, print_info=True) else 1


def del_expire(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    module_option = "{}_videos".format(args.module)

    if not args.expire or config.getint("common", "expire") <= 0:
        return 0

    lifetime = config.getint(module_option, "lifetime")

    if lifetime > 0:
        dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
        expire_path = config.get("common", "hdfs_output", 0, {
            "date": dt_expire,
            "dir": config.get(module_option, "dir")
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
    parser = argparse.ArgumentParser(description='build videos')
    parser.add_argument('--date', dest='date', required=True, help="which date(%%Y-%%m-%%d)")
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    parser.add_argument('--module', dest='module', required=True, choices=["recommendable", "queryable", "recommendable_strict"])
    parser.add_argument('--expire', dest='expire', action="store_true", help="whether to del expired path")

    arguments = parser.parse_args()

    try:
        if not datetime.DateTime(arguments.date).is_perfect_date():
            raise RuntimeError("passed arg [date={}] format error".format(arguments.date))
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
