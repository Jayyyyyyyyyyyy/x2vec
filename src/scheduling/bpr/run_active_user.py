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

    ndays = config.getint("active_user", "n_days")
    assert ndays >= 1, "ndays must be >= 1"

    days_thres = config.getint("active_user", "days_thres")
    clicks_thres = config.getint("active_user", "clicks_thres")

    input_path_list = []
    for dt in datetime.DateTime(args.date).apply_range_by_day(offset=-ndays, exclude=False, reverse=False):
        input_path_list.append(config.get("merge_log", "path", 0, {'date': dt}))
    output_path = config.get("common", "hdfs_output", 0, {'date': args.date, 'dir': config.get("active_user", "dir")})

    if hadoop_shell.exists_all(output_path, flag=True):
        logging.info("output already exists, skip")
        return 0

    if not hadoop_shell.exists_all_with_retry(*input_path_list,
                                              flag=True,
                                              print_info=True,
                                              retry=config.getint("merge_log", "retry"),
                                              interval=config.getint("merge_log", "interval")):
        logging.error("finally, input not ready, exit!")
        return 1

    hadoop_shell.rmr(output_path)

    ss = spark.SparkSubmitWrapper()

    ss.set_master("yarn")\
        .set_deploy_mode("cluster")\
        .set_driver_memory("1G")\
        .set_executor_memory("2G") \
        .add_conf("spark.executor.memoryOverhead", 2048) \
        .set_executor_cores(2)\
        .set_num_executors(100)\
        .add_conf("spark.network.timeout", 600)\
        .set_name(config.get("common", "job_name", 0, {'date': args.date, 'module': "ActiveUser"}))\
        .set_queue(config.get("common", "job_queue"))\
        .set_class("com.td.ml.x2vec.base.ActiveUser")\
        .set_app_jar(FILE_DIR + "/../lib/" + config.get("common", "jar"))\
        .add_app_argument("days_thres", days_thres)\
        .add_app_argument("clicks_thres", clicks_thres)\
        .add_app_argument("input_path", ",".join(input_path_list))\
        .add_app_argument("output_path", output_path)

    return 0 if ss.run(print_cmd=True, print_info=True) else 1


def del_expire(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    if not args.expire or config.getint("common", "expire") <= 0:
        return 0

    for option in ["active_user"]:
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

    arguments = parser.parse_args()

    try:
        if not datetime.DateTime(arguments.date).is_perfect_date():
            raise RuntimeError("passed arg [date={}] format error".format(arguments.date))
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
