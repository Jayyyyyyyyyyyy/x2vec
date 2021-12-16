#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import os
import logging
import argparse
import ConfigParser
import common.datetime_wrapper as datetime
import common.scheduler as scheduler
import common.hdfs_lock as hdfs_lock
import common.hadoop_shell_wrapper as hadoop_shell
import common.fs_wrapper as fs
from common.sys_utils import *
from common.notify import *

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def run_task(args):
    date = args.date_hour.rsplit("-", 1)[0]
    hour = args.date_hour.rsplit("-", 1)[1]
    yesterday = datetime.DateTime(date).apply_offset_by_day(-1)

    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    local_workdir = "{}/../{}".format(
        FILE_DIR,
        config.get(
            "model", "local_dir", 0,
            {
                "date": yesterday,
                "project": config.get("common", "project")
            }
        )
    )
    local_daily_model_succ = os.path.join(local_workdir, "_DEPLOY_SUCCESS")
    logging.info("check daily model succ flag %s", local_daily_model_succ)
    if not fs.exists(local_daily_model_succ):
        logging.error("daily model %s not ready, skip hourly routine!", yesterday)
        return 1, "check_daily_model"

    hdfs_root = config.get("common", "hdfs_root")
    if not hadoop_shell.mkdir(hdfs_root):
        logging.error("fail to mkdir project hdfs root path %s", hdfs_root)
        return 1, "mkdir_root"

    lock = hdfs_lock.HdfsLock(
        config.get("common", "lock"),
        config.getint("common", "lock_interval_hourly"),
        config.getint("common", "lock_timeout_hourly"))
    try:
        if not lock.acquire():
            logging.error("fail to acquire lock, exit")
            return 1, "acquire lock"

        py_bin = config.get("common", "python_bin")

        sched = scheduler.TaskManager()
        sched.bound_threads(config.getint("scheduler", "concurrency"))

        sched.add_task("build_inference", "{} run_build_inference_hourly.py --date {} --hour {} --history_date {} --conf {} --expire".format(
            py_bin, date, hour, yesterday, args.conf
        ), True)
        sched.add_task("build_tfrecords", "{} run_build_tfrecords_hourly.py --date {} --hour {} --model_date {} --conf {} --expire".format(
            py_bin, date, hour, yesterday, args.conf
        ), True)
        sched.add_task("inference", "{} run_inference_hourly.py --date {} --hour {} --model_date {} --conf {}".format(
            py_bin, date, hour, yesterday, args.conf
        ), True)
        sched.add_task("deploy", "{} run_deploy_hourly.py --date {} --hour {} --conf {} --notify --ack --expire".format(
            py_bin, date, hour, args.conf
        ), True)

        sched.set_dependency("build_inference", "build_tfrecords")
        sched.set_dependency("build_tfrecords", "inference")
        sched.set_dependency("inference", "deploy")

        sched.run()
        lock.release()

        if sched.status():
            logging.info("successfully finished all task")
            return 0, None

        return 1, sched.failed_stage()

    except Exception as e:
        logging.exception(e)

    lock.release()
    logging.error("task failed")
    return 1, "exception"


def run(args):
    ret, failed_stage = run_task(args)
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    if ret != 0 and config.getint("notify", "open") > 0:
        phones = config.get("notify", "sms")
        owner = config.get("common", "owner")
        project = config.get("common", "project") + "-hourly"
        if phones:
            msg = "{}例行失败！【失败阶段】{}【Owner】{}【数据】{}【机器】{} 【目录】{}".format(
                project, failed_stage, owner, args.date_hour, get_host_name(), FILE_DIR)
            notify_sms(phones, msg)
        notify_im("{}例行失败！".format(project),
                  date=args.date_hour, stage=failed_stage, owner=owner, host=get_host_name(), path=FILE_DIR)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build samples')
    parser.add_argument('--date_hour', dest='date_hour', required=True, help="which date(%%Y-%%m-%%d)")
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    arguments = parser.parse_args()

    try:
        if not datetime.DateTime(arguments.date_hour).is_perfect_datehour():
            raise RuntimeError("passed arg [date={}] format error".format(arguments.date_hour))
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
