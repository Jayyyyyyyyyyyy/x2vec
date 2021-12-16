#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import os
import logging
import argparse
import ConfigParser
import common.hadoop_shell_wrapper as hadoop_shell
import common.spark_submit_wrapper as spark

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def run(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    input_user_ids = config.get("common", "hdfs_output", 0,
                                {'date': args.train_date, 'dir': config.get("user_ids", "dir")})
    input_path_app_video_dv = config.get("app_video_dv", "path", 0, {'date': args.eval_date})
    output_coverity = config.get("common", "hdfs_output", 0,
                                 {'date': args.train_date, 'dir': config.get("coverity", "dir")})

    hadoop_shell.rmr(output_coverity)

    ss = spark.SparkSubmitWrapper()

    ss.set_master("yarn") \
        .set_deploy_mode("cluster") \
        .set_driver_memory("1G") \
        .set_executor_memory("1G") \
        .set_executor_cores(2) \
        .set_num_executors(50) \
        .add_conf("spark.network.timeout", 600) \
        .set_name(config.get("common", "job_name", 0, {'date': args.train_date, 'module': "CalcCoverity"})) \
        .set_queue(config.get("common", "job_queue")) \
        .set_class("com.td.ml.x2vec.bpr.CalcCoverity") \
        .set_app_jar(FILE_DIR + "/../lib/" + config.get("common", "jar")) \
        .add_app_argument("input_path_user_ids", input_user_ids) \
        .add_app_argument("input_path_app_video_dv", input_path_app_video_dv) \
        .add_app_argument("output_path_coverity", output_coverity)

    return 0 if ss.run(print_cmd=True, print_info=True) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build samples')
    parser.add_argument('--train_date', dest='train_date', required=True, help="which date(%%Y-%%m-%%d)")
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    parser.add_argument('--eval_date', dest='eval_date', required=True, help="which date(%%Y-%%m-%%d)")
    arguments = parser.parse_args()

    try:
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)