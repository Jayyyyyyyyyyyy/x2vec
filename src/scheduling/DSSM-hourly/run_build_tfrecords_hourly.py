#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
sys.path.append("../DSSM")
import os
import logging
import argparse
import ConfigParser
import common.datetime_wrapper as datetime
import common.hadoop_shell_wrapper as hadoop_shell
from common.hadoop_streaming_wrapper import HadoopStreamingWrapper
from model_configuration import ModelConfiguration
import common.fs_wrapper as fs

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def process_build_samples(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    local_train_path = "{}/../{}".format(
        FILE_DIR,
        config.get(
            "model",
            "local_dir",
            0,
            {
                "date": args.model_date,
                "project": config.get("common", "project")
            }
        )
    )

    local_train_succ_file = os.path.join(local_train_path, "_TRAIN_SUCCESS")
    local_deploy_succ_file = os.path.join(local_train_path, "_DEPLOY_SUCCESS")
    model_conf_file = os.path.join(local_train_path, "model.conf")

    if not fs.exists(local_train_succ_file) or not fs.exists(local_deploy_succ_file):
        logging.error("model succ flag not found")
        return 1

    if not fs.exists(model_conf_file):
        logging.error("model conf not found")
        return
    model_conf = ModelConfiguration().load(model_conf_file)
    if model_conf is None or not model_conf.check():
        logging.error("fail to build model configuration")
        return 1
    model_conf.show()

    input_path = config.get("common", "hdfs_output_hourly", 0, {
        "date": args.date,
        "hour": args.hour,
        "dir": config.get("inference_sample_hourly", "raw_dir")
    })
    output_path = config.get("common", "hdfs_output_hourly", 0, {
        "date": args.date,
        "hour": args.hour,
        "dir": config.get("inference_sample_hourly", "tfrecord_dir")
    })
    output_path_temp = output_path + ".temp"

    if hadoop_shell.exists_all(
            output_path,
            flag=True):
        logging.info("output already exists, skip")
        return 0

    if not hadoop_shell.exists_all(input_path,
                                   flag=True,
                                   print_missing=True):
        logging.error("input not ready, exit!")
        return 1

    if not hadoop_shell.rmr(output_path, output_path_temp):
        logging.error("fail to clear output folder")
        return 1

    hadoop_shell.mkdir(output_path)

    streaming_proc = HadoopStreamingWrapper()
    streaming_proc.add_generic_option(
        "mapred.job.name",
        config.get("common", "job_name", 0, {
            'date': args.date + "-" + args.hour,
            'module': "BuildInferenceTFRecordsHourly"
        })
    )
    reduce_tasks = config.getint("inference_sample_hourly", "tfrecord_partitions")
    streaming_proc.add_generic_option("mapred.map.tasks", 1000)
    streaming_proc.add_generic_option("mapred.reduce.tasks", reduce_tasks)
    streaming_proc.add_generic_option("mapred.job.map.capacity", 300)
    streaming_proc.add_generic_option("mapred.job.reduce.capacity", 0)
    streaming_proc.add_generic_option("mapred.job.priority", "VERY_HIGH")
    streaming_proc.add_generic_option("mapred.map.over.capacity.allowed", "false")
    streaming_proc.add_generic_option("mapred.reduce.over.capacity.allowed", "false")
    streaming_proc.add_generic_option("stream.memory.limit", 2048)
    streaming_proc.add_generic_option("mapreduce.map.memory.mb", 1024)
    streaming_proc.add_generic_option("mapreduce.reduce.memory.mb", 2048)
    streaming_proc.add_generic_option("mapred.max.map.failures.percent", 5)
    streaming_proc.add_generic_option("mapred.max.reduce.failures.percent", 5)
    streaming_proc.add_generic_option("mapreduce.job.reduce.slowstart.completedmaps", 0.99)
    streaming_proc.add_generic_option("mapred.hce.replace.streaming", "false")
    streaming_proc.add_generic_option("mapreduce.job.queuename", config.get("common", "job_queue"))
    streaming_proc.add_generic_option("mapreduce.map.sort.spill.percent", 0.8)
    streaming_proc.add_streaming_option("input", input_path)
    streaming_proc.add_streaming_option("output", output_path_temp)
    streaming_proc.add_streaming_option(
        "mapper",
        "./pyenv/pyenv/bin/python2.7 mapred_build_tfrecords.py --stage {}".format("map")
    )
    streaming_proc.add_streaming_option(
        "reducer",
        "./pyenv/pyenv/bin/python2.7 mapred_build_tfrecords.py --stage {} --batch_size {} --vocab_vid_size {} --output_path {} {}".format(
            "reduce",
            model_conf.batch_size,
            model_conf.vocab_vid_size,
            output_path,
            "--keep_diu"
        )
    )
    streaming_proc.add_streaming_option("file", "../DSSM/mapred_build_tfrecords.py")
    streaming_proc.add_streaming_option("file", "../common/reservoir_sampling.py")
    streaming_proc.add_streaming_option("file", "../common/shell_wrapper.py")
    streaming_proc.add_streaming_option("file", "../common/hadoop_shell_wrapper.py")
    streaming_proc.add_streaming_option("file", "../common/sparse_values.py")
    streaming_proc.add_streaming_option("file", "../common/datetime_wrapper.py")
    streaming_proc.add_streaming_option("cacheArchive", "%s#pyenv" % config.get("common", "pyenv"))

    ret = streaming_proc.run(print_cmd=True, print_info=True)

    if ret:
        hadoop_shell.touchz(output_path + "/_SUCCESS", print_cmd=True)

    hadoop_shell.rmr(output_path_temp)
    return 0 if ret else 1


def del_expire(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    if not args.expire or config.getint("common", "expire") <= 0:
        return 0

    lifetime = config.getint("inference_sample_hourly", "lifetime")

    if lifetime > 0:
        dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
        expire_path = config.get("common", "hdfs_output", 0, {
            "date": dt_expire,
            "hour": "*",
            "dir": config.get("inference_sample_hourly", "tfrecord_dir")
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
    parser = argparse.ArgumentParser(description='build samples')
    parser.add_argument('--date', dest='date', required=True, help="which date(%%Y-%%m-%%d)")
    parser.add_argument('--hour', dest='hour', required=True, help="which hour(%%Y-%%m-%%d)")
    parser.add_argument('--model_date', dest='model_date', required=True, help="which model_date(%%Y-%%m-%%d)")
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    parser.add_argument('--expire', dest='expire', action="store_true", help="whether to del expired path")

    arguments = parser.parse_args()

    try:
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
