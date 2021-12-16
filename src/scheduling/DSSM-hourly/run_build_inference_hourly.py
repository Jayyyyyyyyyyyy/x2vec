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

    ndays = config.getint("inference_sample", "n_days")
    assert ndays >= 1, "ndays must be >= 1"

    ts_bound = datetime.DateTime.to_ts(datetime.DateTime(args.history_date).apply_offset_by_day(-ndays))
    seq_max_size = config.getint("model", "seq_max_size")
    seq_min_size = config.getint("inference_sample", "seq_min_size")
    d_lastest = config.getint("inference_sample", "d_lastest")

    input_path_user_history = config.get("user_history", "path", 0, {'date': args.history_date})
    input_path_vocab_vid = config.get("common", "hdfs_output", 0, {
        "date": args.history_date,
        "dir": config.get("vocab", "vid_dir")
    })

    input_path_video_play_speed = config.get("video_play_speed", "path", 0, {"date": args.date})

    output_path = config.get("common", "hdfs_output_hourly", 0, {
        "date": args.date,
        "hour": args.hour,
        "dir": config.get("inference_sample_hourly", "raw_dir")
    })

    if hadoop_shell.exists_all(output_path, flag=True, print_missing=True):
        logging.info("output already exists, skip")
        return 0

    if not hadoop_shell.exists_all_with_retry(
            input_path_user_history,
            input_path_vocab_vid,
            flag=True,
            print_info=True,
            retry=config.getint("common", "upstream_retry"),
            interval=config.getint("common", "upstream_interval")):
        logging.error("finally, input not ready, exit!")
        return 1
    if not hadoop_shell.exists_all(input_path_video_play_speed, flag=False, print_cmd=True, print_missing=True):
        logging.error("finally, input %s not ready, exit!", input_path_video_play_speed)
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
        .set_name(config.get("common", "job_name", 0, {'date': args.date + "-" + args.hour, 'module': "BuildInferenceHourly"}))\
        .set_queue(config.get("common", "job_queue"))\
        .set_class("com.td.ml.x2vec.DSSM.BuildInferenceHourly")\
        .set_app_jar(FILE_DIR + "/../lib/" + config.get("common", "jar"))\
        .add_app_argument("ts_bound", ts_bound)\
        .add_app_argument("seq_max_size", seq_max_size)\
        .add_app_argument("seq_min_size", seq_min_size)\
        .add_app_argument("d_lastest", d_lastest)\
        .add_app_argument("input_path_user_history", input_path_user_history)\
        .add_app_argument("input_path_vocab_vid", input_path_vocab_vid)\
        .add_app_argument("input_path_video_play_speed", input_path_video_play_speed + "/*") \
        .add_app_argument("output_path", output_path)

    return 0 if ss.run(print_cmd=True, print_info=True) else 1


def del_expire(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    if not args.expire or config.getint("common", "expire") <= 0:
        return 0

    lifetime = config.getint("inference_sample_hourly", "lifetime")

    if lifetime > 0:
        dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
        expire_path = config.get("common", "hdfs_output_hourly", 0, {
            "date": dt_expire,
            "hour": "*",
            "dir": config.get("inference_sample_hourly", "raw_dir")
        })
        if not hadoop_shell.rmr(expire_path):
            logging.error("fail to del expired path: %s", expire_path)
            return 1
    return 0


def run(args):
    if process_build_samples(args) == 0 and del_expire(args) == 0:
        return 0
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build user inference samples')
    parser.add_argument('--date', dest='date', required=True, help="which date(%%Y-%%m-%%d)")
    parser.add_argument('--hour', dest='hour', required=True, help="which hour(%%Y-%%m-%%d)")
    parser.add_argument('--history_date', dest='history_date', required=True, help="which date(%%Y-%%m-%%d)")
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
