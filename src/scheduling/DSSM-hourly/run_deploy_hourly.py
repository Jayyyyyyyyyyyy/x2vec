#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import os
import glob
import logging
import argparse
import ConfigParser
import common.datetime_wrapper as datetime
import common.hadoop_shell_wrapper as hadoop_shell
import common.fs_wrapper as fs
import common.shell_wrapper as shell
import common.spark_submit_wrapper as spark
from common.zookeeper_utils import *
from collections import namedtuple
import time

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


ModelInfo = namedtuple("ModelInfo", ["hdfs", "redis"])


def parse_model_info(info_str):
    try:
        if info_str is None or len(info_str) == 0:
            return None
        kvs = info_str.split("&")
        params = dict()
        for kv in kvs:
            k, v = kv.split("=", 1)
            if k != "" and v != "":
                params[k] = v
        return ModelInfo(hdfs=params.get("hdfs"), redis=params.get("redis"))
    except Exception as e:
        print >> sys.stderr, e
    return None


def format_model_info(model_info):
    return "hdfs={}&redis={}".format(model_info.hdfs, model_info.redis)


def fetch_synchronized_redis(config, args):
    zk_host = config.get("zookeeper", "host")
    zk_u2i_ack_node = config.get("zookeeper", "u2i_ack_node")

    # 连接zookeeper
    zk = zk_connect(zk_host)
    if zk is None:
        logging.error("fail to connect zookeeper %s", zk_host)
        return -1, None

    # 获取在线服务一致使用的redis
    server_redis_list = []

    u2i_server_ack_nodes = zk_get_children(zk, zk_u2i_ack_node)
    logging.info("u2i servers: %s", u2i_server_ack_nodes)
    if u2i_server_ack_nodes is not None:
        for server_ack_node in u2i_server_ack_nodes:
            server_node_path = "{}/{}".format(zk_u2i_ack_node, server_ack_node)
            server_model_str = zk_get(zk, server_node_path)
            server_model_info = parse_model_info(server_model_str)
            if server_model_info is None:
                logging.error("fail to parse u2i server [%s] ack info [%s]", server_node_path, server_model_str)
                return -1, None
            logging.info("u2i server=%s, model hdfs=%s, redis=%s",
                         server_node_path, server_model_info.hdfs, server_model_info.redis)
            server_redis_list.append(server_model_info.redis)

    server_redis_list = list(set(server_redis_list))
    if len(server_redis_list) == 0:
        return 0, None
    if len(server_redis_list) > 1:
        logging.error("all server loaded models are not synchronized")
        return -1, None
    return 0, server_redis_list[0]


def fetch_working_redis(config, args):
    """
    根据zookeeper中当前servers统一的配置，获取当前工作的redis
    :param config: 配置文件对象
    :param args: 命令行参数
    :return: 闲置redis host（失败：None）
    """

    ret, active_redis = fetch_synchronized_redis(config, args)
    if ret != 0 or not active_redis:
        logging.error("fail to fetch synchronized redis")
        return -1, None

    return 0, active_redis


def deploy_redis(config, args, hdfs_model_file):
    ret, working_redis = fetch_working_redis(config, args)
    if ret != 0 or not working_redis:
        logging.error("fail to get working redis host, exit")
        return 1

    ss = spark.SparkSubmitWrapper()
    ss.set_master("yarn") \
        .set_deploy_mode("cluster") \
        .set_driver_memory("1G") \
        .set_executor_memory("1G") \
        .set_executor_cores(1) \
        .set_num_executors(50) \
        .add_conf("spark.network.timeout", 600) \
        .set_name(config.get("common", "job_name", 0, {'date': args.date, 'module': "FeedRedisHourly"})) \
        .set_queue(config.get("common", "job_queue")) \
        .set_class("com.td.ml.x2vec.base.FeedRedis") \
        .set_app_jar(FILE_DIR + "/../lib/" + config.get("common", "jar")) \
        .add_app_argument("input_path", hdfs_model_file) \
        .add_app_argument("redis", working_redis) \
        .add_app_argument("dim", config.getint("model", "embed_vid_size")) \
        .add_app_argument("concurrency", config.getint("feed_redis", "concurrency")) \
        .add_app_argument("flushall", "false")

    if ss.run(print_cmd=True, print_info=True):
        logging.info("successfully feed user emb into redis")
        return 0
    logging.error("fail to feed user emb into redis")
    return 1


def remove_expire(config, args):
    # 删除过期
    if args.expire and config.getint("common", "expire") > 0:
        lifetime = config.getint("model", "hdfs_lifetime")
        if lifetime > 0:
            dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
            expire_path = config.get(
                "common", "hdfs_output_hourly", 0,
                {'date': dt_expire, "hour": "*", 'dir': config.get("model", "hdfs_dir")})
            logging.info("removing hdfs path %s", expire_path)
            if fs.remove_dir(expire_path):
                logging.info("successfully removed hdfs expire %s", expire_path)
            else:
                logging.warning("failed to remove hdfs expire %s", expire_path)

        lifetime = config.getint("model", "local_lifetime")
        if lifetime > 0:
            dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
            expire_path = "{}/../{}".format(
                FILE_DIR,
                config.get("model", "local_dir", 0,
                           {
                               "date": dt_expire,
                               "project": config.get("common", "project") + "-hourly"
                           }))
            for x in glob.glob(expire_path + "-*"):
                logging.info("removing local path %s", x)
                if fs.remove_dir(x):
                    logging.info("successfully removed local expire %s", x)
                else:
                    logging.warning("failed to remove local expire %s", x)
                    return 1
    else:
        logging.info("expire switch close, skip removing expired data")

    return 0


def run(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    local_workdir = "{}/../{}".format(
        FILE_DIR,
        config.get("model", "local_dir", 0,
                   {
                       "date": args.date + "-" + args.hour,
                       "project": config.get("common", "project") + "-hourly"
                   }))
    local_user_model_file = os.path.join(local_workdir, "user.emb")
    local_inference_succ_flag = os.path.join(local_workdir, "_INFERENCE_SUCCESS")
    local_deploy_succ_file = os.path.join(local_workdir, "_DEPLOY_SUCCESS")

    hdfs_model_path = config.get(
        "common", "hdfs_output_hourly", 0,
        {'date': args.date, "hour": args.hour, 'dir': config.get("model", "hdfs_dir") + "-hourly"})
    hdfs_model_file = hdfs_model_path + "/user.emb"

    if fs.exists(local_deploy_succ_file):
        logging.info("deploy succ flag found! skip")
        return 0

    # 判断向量文件是否存在
    if not fs.exists(local_inference_succ_flag) or not fs.exists(local_user_model_file):
        logging.error("model not found complete, exit")
        return 1

    # 清理hdfs的模型目录
    if not hadoop_shell.rmr(hdfs_model_path) or not hadoop_shell.mkdir(hdfs_model_path):
        logging.error("fail to clear hdfs model path: %s", hdfs_model_path)
        return 1

    logging.info("===== start to upload user embedding to hdfs =====")
    # 上传用户/视频向量文件
    if not hadoop_shell.put(local_user_model_file, hdfs_model_path):
        logging.error("fail to put embedding to hdfs %s", hdfs_model_path)
        return 1

    logging.info("===== start to feed user embedding into redis =====")
    # 把用户向量灌入备份redis
    if deploy_redis(config, args, hdfs_model_file) != 0:
        logging.error("fail to import user embedding to redis")
        return 1

    logging.info("deploy hourly %s-%s succ", args.date, args.hour)

    logging.info("===== start to remove expire data =====")
    # 删除过期本地模型目录和hdfs目录
    if remove_expire(config, args) != 0:
        logging.error("fail to remove expire")
        return 1

    fs.touch(local_deploy_succ_file)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deploy and backup model')
    parser.add_argument('--date', dest='date', required=True, help="which date(%%Y-%%m-%%d)")
    parser.add_argument('--hour', dest='hour', required=True, help="which hour(%%Y-%%m-%%d)")
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    parser.add_argument('--notify', dest='notify', action="store_true", help="notify zookeeper")
    parser.add_argument("--ack", dest="ack", action="store_true", help="wait for sync servers after deploy")
    parser.add_argument('--expire', dest='expire', action="store_true", help="whether to del expired path")
    arguments = parser.parse_args()

    try:
        if not datetime.DateTime(arguments.date).is_perfect_date():
            raise RuntimeError("passed arg [date={}] format error".format(arguments.date))
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
