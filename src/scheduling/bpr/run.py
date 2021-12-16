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
from common.sys_utils import *
from common.notify import *

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def run_task(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    lock = hdfs_lock.HdfsLock(
        config.get("common", "lock"),
        config.getint("common", "lock_interval"),
        config.getint("common", "lock_timeout"))
    try:
        if not lock.acquire():
            logging.error("fail to acquire lock, exit")
            return 1, "acquire lock"

        py_bin = config.get("common", "python_bin")

        sched = scheduler.TaskManager()
        sched.bound_threads(config.getint("scheduler", "concurrency"))

        sched.add_task("build_samples", "{} run_build_sample.py --date {} --conf {} --expire".format(
            py_bin, args.date, args.conf
        ), True)
        # sched.add_task("active_user", "{} run_active_user.py --date {} --conf {} --expire".format(
        #     py_bin, args.date, args.conf
        # ), True)
        sched.add_task("train", "{} run_train.py --date {} --conf {} --expire".format(
            py_bin, args.date, args.conf
        ), True)
        sched.add_task("deploy", "{} run_deploy.py --date {} --conf {} --expire --notify --ack --disable_u2u".format(
            py_bin, args.date, args.conf
        ), True)

        sched.set_dependency("build_samples", "train")
        sched.set_dependency("train", "deploy")
        # sched.set_dependency("active_user", "deploy")

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
        project = config.get("common", "project")
        if phones:
            msg = "{}例行失败！【失败阶段】{}【Owner】{}【数据】{}【机器】{} 【目录】{}".format(
                project, failed_stage, owner, args.date, get_host_name(), FILE_DIR)
            notify_sms(phones, msg)
        notify_im("{}例行失败！".format(project),
                  date=args.date, stage=failed_stage, owner=owner, host=get_host_name(), path=FILE_DIR)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build samples')
    parser.add_argument('--date', dest='date', required=True, help="which date(%%Y-%%m-%%d)")
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    arguments = parser.parse_args()

    try:
        if not datetime.DateTime(arguments.date).is_perfect_date():
            raise RuntimeError("passed arg [date={}] format error".format(arguments.date))
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
