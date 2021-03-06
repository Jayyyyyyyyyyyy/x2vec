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
import common.hadoop_shell_wrapper as hadoop_shell
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

    hdfs_root = config.get("common", "hdfs_root")
    if not hadoop_shell.mkdir(hdfs_root):
        logging.error("fail to mkdir project hdfs root path %s", hdfs_root)
        return 1

    try:
        py_bin = config.get("common", "python_bin")

        sched = scheduler.TaskManager()
        sched.bound_threads(config.getint("scheduler", "concurrency"))

        sched.add_task("queryable_videos", "{} run_build_videos.py --date {} --conf {} --module queryable --expire".format(
            py_bin, args.date, args.conf
        ), True)
        sched.add_task("recommendable_videos", "{} run_build_videos.py --date {} --conf {} --module recommendable --expire".format(
            py_bin, args.date, args.conf
        ), True)
        sched.add_task("recommendable_strict_videos", "{} run_build_videos.py --date {} --conf {} --module recommendable_strict --expire".format(
            py_bin, args.date, args.conf
        ), True)

        sched.run()

        if sched.status():
            logging.info("successfully finished all task")
            return 0, None

        return 1, sched.failed_stage()

    except Exception as e:
        logging.exception(e)

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
            msg = "{}?????????????????????????????????{}???Owner???{}????????????{}????????????{} ????????????{}".format(
                project, failed_stage, owner, args.date, get_host_name(), FILE_DIR)
            notify_sms(phones, msg)
        notify_im("{}???????????????".format(project),
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
