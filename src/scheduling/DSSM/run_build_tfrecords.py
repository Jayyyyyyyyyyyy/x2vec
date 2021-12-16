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
from common.hadoop_streaming_wrapper import HadoopStreamingWrapper
from model_configuration import ModelConfiguration
import common.fs_wrapper as fs

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def extract_vocab_size(vocab_filename):
    max_fid = -1
    with open(vocab_filename, "r") as fin:
        for line in fin:
            fid, fname = line.strip().split("\t")[:2]
            max_fid = max(max_fid, int(fid))
    assert max_fid > 1
    return max_fid + 1


def build_model_conf(args, config):
    local_model_path = "{}/../{}".format(
        FILE_DIR,
        config.get(
            "model",
            "local_dir",
            0,
            {
                "date": args.date,
                "project": config.get("common", "project")
            }
        )
    )

    model_conf_file = os.path.join(local_model_path, "model.conf")
    if fs.exists(model_conf_file):
        return ModelConfiguration().load(model_conf_file)

    if not fs.remove_dir(local_model_path) or not fs.mkdir(local_model_path):
        logging.error("fail to clear %s", local_model_path)
        return None

    vocab_vid_file = os.path.join(local_model_path, "vid.vocab")

    vocab_vid_hdfs = config.get("common", "hdfs_output", 0, {
        "date": args.date,
        "dir": config.get("vocab", "vid_dir")
    })
    if not hadoop_shell.exists_all(vocab_vid_hdfs, flag=True, print_missing=True):
        logging.error("vocab file not ready, exit")
        return None

    if not fs.exists(vocab_vid_file) and not hadoop_shell.getmerge(vocab_vid_hdfs, vocab_vid_file):
        logging.error("fail to download %s", vocab_vid_hdfs)
        return None

    model_conf = ModelConfiguration()
    model_conf.batch_size = config.getint("model", "batch_size")
    model_conf.embed_vid_size = config.getint("model", "embed_vid_size")
    model_conf.vocab_vid_size = extract_vocab_size(vocab_vid_file)
    model_conf.seq_max_size = config.getint("model", "seq_max_size")
    model_conf.epoches = config.getint("model", "epoches")
    model_conf.gpu_core = config.getint("model", "gpu_core")
    model_conf.negative_samples = config.getint("model", "negative_samples")
    model_conf.l2_reg_rate = config.getfloat("model", "l2_reg_rate")
    model_conf.model_keep = config.getint("model", "model_keep")
    model_conf.checkpoint_steps = config.getint("model", "checkpoint_steps")
    model_conf.learning_rate = config.getfloat("model", "learning_rate")
    model_conf.fc_layers = eval(config.get("model", "fc_layers"))
    model_conf.triplet_margin = config.getfloat("model", "triplet_margin")
    model_conf.eval_step = config.getint("model", "eval_step")
    model_conf.test_fraction = config.getfloat("test_sample", "test_fraction")
    model_conf.persist(model_conf_file)
    return model_conf


def process_build_samples(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    stage_option = "{}_sample".format(args.stage)

    model_conf = build_model_conf(args, config)
    if model_conf is None or not model_conf.check():
        logging.error("fail to build model configuration")
        return 1
    model_conf.show()

    input_path_sample_raw = config.get("common", "hdfs_output", 0, {
        "date": args.date,
        "dir": config.get(stage_option, "raw_dir")
    })
    output_path_sample_tfrecord = config.get("common", "hdfs_output", 0, {
        "date": args.date,
        "dir": config.get(stage_option, "tfrecord_dir")
    })
    output_path_sample_tfrecord_temp = output_path_sample_tfrecord + ".temp"

    if hadoop_shell.exists_all(
            output_path_sample_tfrecord,
            flag=True):
        logging.info("output already exists, skip")
        return 0

    if not hadoop_shell.exists_all(input_path_sample_raw,
                                   flag=True,
                                   print_missing=True):
        logging.error("input not ready, exit!")
        return 1

    if not hadoop_shell.rmr(output_path_sample_tfrecord, output_path_sample_tfrecord_temp):
        logging.error("fail to clear output folder")
        return 1

    hadoop_shell.mkdir(output_path_sample_tfrecord)

    streaming_proc = HadoopStreamingWrapper()
    streaming_proc.add_generic_option(
        "mapred.job.name",
        config.get("common", "job_name", 0, {
            'date': args.date,
            'module': "BuildTFRecords.{}".format(args.stage)
        })
    )
    reduce_tasks = config.getint(stage_option, "tfrecord_partitions")
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
    streaming_proc.add_streaming_option("input", input_path_sample_raw)
    streaming_proc.add_streaming_option("output", output_path_sample_tfrecord_temp)
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
            output_path_sample_tfrecord,
            "--keep_diu" if args.stage == "inference" else ""
        )
    )
    streaming_proc.add_streaming_option("file", "./mapred_build_tfrecords.py")
    streaming_proc.add_streaming_option("file", "../common/reservoir_sampling.py")
    streaming_proc.add_streaming_option("file", "../common/shell_wrapper.py")
    streaming_proc.add_streaming_option("file", "../common/hadoop_shell_wrapper.py")
    streaming_proc.add_streaming_option("file", "../common/sparse_values.py")
    streaming_proc.add_streaming_option("file", "../common/datetime_wrapper.py")
    streaming_proc.add_streaming_option("cacheArchive", "%s#pyenv" % config.get("common", "pyenv"))

    ret = streaming_proc.run(print_cmd=True, print_info=True)

    if ret:
        hadoop_shell.touchz(output_path_sample_tfrecord + "/_SUCCESS", print_cmd=True)

    hadoop_shell.rmr(output_path_sample_tfrecord_temp)
    return 0 if ret else 1


def del_expire(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    if not args.expire or config.getint("common", "expire") <= 0:
        return 0

    stage_option = "{}_sample".format(args.stage)

    lifetime = config.getint(stage_option, "lifetime")

    if lifetime > 0:
        dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
        expire_path = config.get("common", "hdfs_output", 0, {
            "date": dt_expire,
            "dir": config.get(stage_option, "tfrecord_dir")
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
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    parser.add_argument('--expire', dest='expire', action="store_true", help="whether to del expired path")
    parser.add_argument('--stage', dest='stage', required=True, choices=["train", "test", "inference"], help="train/test/inference")

    arguments = parser.parse_args()

    try:
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
