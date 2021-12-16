#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../../")
import os
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


def check_synchronized(config, zk_ack_node, model_info):
    zk_host = config.get("zookeeper", "host")
    # 连接zookeeper
    zk = zk_connect(zk_host)
    if zk is None:
        logging.error("fail to connect zookeeper %s", zk_host)
        return -1

    # 获取在线服务的ack nodes
    server_ack_nodes = zk_get_children(zk, zk_ack_node)
    logging.info("servers: %s", server_ack_nodes)

    if server_ack_nodes is not None and len(server_ack_nodes) > 0:
        for server_ack_node in server_ack_nodes:
            server_node_path = "{}/{}".format(zk_ack_node, server_ack_node)
            server_model_str = zk_get(zk, server_node_path)
            server_model_info = parse_model_info(server_model_str)
            if server_model_info is None:
                logging.error("fail to parse server [%s] ack info [%s]", server_node_path, server_model_str)
                return -1
            logging.info("server=%s, model hdfs=%s, redis=%s",
                         server_node_path, server_model_info.hdfs, server_model_info.redis)
            if server_model_info.redis != model_info.redis or server_model_info.hdfs != model_info.hdfs:
                logging.error("server [%s] not synchronized", server_node_path)
                return -1
    return 0


def deploy_redis(config, args):

    input_path = config.get(
        "common", "hdfs_output", 0,
        {'date': args.date, 'dir': config.get("queryable_embedding", "dir")})
    target_redis = "{}:{}".format(
        config.get("feed_redis", "redis_host"),
        config.getint("feed_redis", "redis_port")
    )
    version = config.get("bert", "version")
    ttl = config.getint("feed_redis", "ttl")
    flushall = config.get("feed_redis", "flushall")
    concurrency = config.getint("feed_redis", "concurrency")

    logging.info("===== start to feed embedding into redis %s (ttl = %d, version = %s, flushall = %s, concurrency = %d)",
                 target_redis, ttl, version, flushall, concurrency)
    ss = spark.SparkSubmitWrapper()
    ss.set_master("yarn") \
        .set_deploy_mode("cluster") \
        .set_driver_memory("1G") \
        .set_executor_memory("1G") \
        .set_executor_cores(1) \
        .set_num_executors(50) \
        .add_conf("spark.network.timeout", 600) \
        .set_name(config.get("common", "job_name", 0, {'date': args.date, 'module': "FeedRedis"})) \
        .set_queue(config.get("common", "job_queue")) \
        .set_class("com.td.ml.x2vec.base.FeedRedis") \
        .set_app_jar(FILE_DIR + "/../../lib/" + config.get("common", "jar")) \
        .add_app_argument("input_path", input_path) \
        .add_app_argument("redis", target_redis) \
        .add_app_argument("version", version) \
        .add_app_argument("dim", config.getint("bert", "dim")) \
        .add_app_argument("ttl", ttl)\
        .add_app_argument("concurrency", concurrency) \
        .add_app_argument("flushall", flushall)

    if ss.run(print_cmd=True, print_info=True):
        logging.info("successfully feed emb into redis")
        return 0
    logging.error("fail to feed emb into redis")
    return 1


def generate_index(config, args):
    local_train_path = "{}/../../{}".format(
        FILE_DIR,
        config.get("faiss_index", "local_dir", 0,
                   {
                       "date": args.date,
                       "project": config.get("common", "project")
                   }
                   ))

    local_item_model_file = os.path.join(local_train_path, "item.emb")
    hdfs_item_model_path = config.get("common", "hdfs_output", 0, {
        "date": args.date,
        "dir": config.get("recommendable_embedding", "dir")
    })
    version = config.get("bert", "version")

    if not fs.remove_file(local_item_model_file) or \
            not hadoop_shell.getmerge(hdfs_item_model_path, local_item_model_file):
        logging.error("fail to download index embedding")
        return 1

    cmd = "{}/../../bin/build_faiss_index " \
          "--clusters {} " \
          "--dim {} " \
          "--metric {} " \
          "--samples {} " \
          "--inputs {} " \
          "--output {} " \
          "--version {}".format(FILE_DIR,
                                config.getint("faiss", "clusters"),
                                config.getint("bert", "dim"),
                                config.get("faiss", "metric"),
                                config.getint("faiss", "samples"),
                                local_item_model_file,
                                local_train_path,
                                version)

    logging.info("build_index cmd: %s", cmd)
    if not shell.shell_command(cmd=cmd, print_info=True):
        logging.error("build_index %s failed", args.date)
        return 1
    logging.info("successfully build_index %s", args.date)
    fs.remove_file(local_item_model_file)
    # =============  结束：对检索向量建faiss索引
    return 0


def notify_zookeeper(config, args, model_info):
    if not args.notify:
        logging.info("notify switch closed, skip notifying zookeeper")
        return 0

    if not isinstance(model_info, ModelInfo) or \
            len(model_info.redis) == 0 or \
            len(model_info.hdfs) == 0:
        logging.error("error u2i model_info, exit notify_zookeeper")
        return 1

    # 连接zookeeper
    zk_host = config.get("zookeeper", "host")
    zk = zk_connect(zk_host)
    if zk is None:
        logging.error("fail to connect zookeeper %s", zk_host)
        return 1

    zk_model_node = config.get("zookeeper", "model_node")
    zk_ack_node = config.get("zookeeper", "ack_node")

    if not zk_create(zk, zk_model_node) or not zk_create(zk, zk_ack_node):
        logging.error("fail to create zookeeper model node %s", zk_model_node)
        return 1
    updated_model_str = format_model_info(model_info)
    logging.info("start to set zookeeper node %s value=[%s]", zk_model_node, updated_model_str)
    if not zk_set(zk, zk_model_node, updated_model_str):
        logging.error("fail to update zookeeper model node %s", zk_model_node)
        return 1

    # =============  开始：检查servers是否成功加载新模型
    if args.ack:
        logging.info("start checking whether servers have loaded new models")
        attempt = 1
        ack_retry = config.getint("common", "ack_retry")
        ack_interval = config.getint("common", "ack_interval")
        zk_ack_node = config.get("zookeeper", "ack_node")
        while attempt <= ack_retry:
            logging.info("checking synchronized attempt %d of %d", attempt, ack_retry)
            time.sleep(ack_interval)
            if check_synchronized(config, zk_ack_node, model_info) == 0:
                logging.info("all models have loaded new models")
                break
            attempt += 1

        if attempt > ack_retry:
            logging.error("not all models have loaded new models, failed")
            return 1

    return 0


def remove_expire(config, args):
    # 删除过期
    if args.expire and config.getint("common", "expire") > 0:
        lifetime = config.getint("faiss_index", "hdfs_lifetime")
        if lifetime > 0:
            dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
            expire_path = config.get("common", "hdfs_output", 0,
                                     {'date': dt_expire, 'dir': config.get("faiss_index", "hdfs_dir")})
            if hadoop_shell.rmr(expire_path):
                logging.info("successfully removed hdfs expire %s", expire_path)
            else:
                logging.warning("failed to remove hdfs expire %s", expire_path)
                return 1

        lifetime = config.getint("faiss_index", "local_lifetime")
        if lifetime > 0:
            dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
            expire_path = "{}/../../{}".format(
                FILE_DIR,
                config.get("faiss_index", "local_dir", 0,
                           {
                               "date": dt_expire,
                               "project": config.get("common", "project")
                           }))
            if fs.remove_dir(expire_path):
                logging.info("successfully removed local expire %s", expire_path)
            else:
                logging.warning("failed to remove local expire %s", expire_path)
    else:
        logging.info("expire switch close, skip removing expired data")

    return 0


def run(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    local_model_path = "{}/../../{}".format(
        FILE_DIR,
        config.get("faiss_index", "local_dir", 0,
                   {
                       "date": args.date,
                       "project": config.get("common", "project")
                   }))

    hdfs_model_path = config.get("common", "hdfs_output", 0,
                                 {'date': args.date, 'dir': config.get("faiss_index", "hdfs_dir")})

    target_redis = "{}:{}".format(
        config.get("feed_redis", "redis_host"),
        config.getint("feed_redis", "redis_port")
    )

    local_deploy_succ_file = os.path.join(local_model_path, "_DEPLOY_SUCCESS")

    if fs.exists(local_deploy_succ_file):
        logging.info("deploy succ flag found! skip")
        return 0

    logging.info("===== clear local model path = %s =====", local_model_path)
    if not fs.remove_dir(local_model_path) or not fs.mkdir(local_model_path):
        logging.error("fail to clear local ")

    logging.info("===== clear hdfs model path = %s =====", hdfs_model_path)
    if not hadoop_shell.rmr(hdfs_model_path) or not hadoop_shell.mkdir(hdfs_model_path):
        logging.error("fail to clear hdfs model path: %s", hdfs_model_path)
        return 1

    logging.info("===== start to feed query embedding into redis =====")
    if deploy_redis(config, args) != 0:
        logging.error("fail to import embedding to redis")
        return 1

    logging.info("===== start to generate faiss index =====")
    # 生成视频向量的faiss索引
    if generate_index(config, args) != 0:
        logging.error("failed generate_u2i_index")
        return 1

    logging.info("===== start to upload faiss index =====")
    # 把模型上传至hdfs，server会从hdfs拉取模型
    if not hadoop_shell.put(local_model_path + "/*", hdfs_model_path):
        logging.error("fail to put index to hdfs %s", hdfs_model_path)
        return 1

    # 通知zookeeper更新模型

    model_info = ModelInfo(hdfs=hdfs_model_path, redis=target_redis)
    if notify_zookeeper(config, args, model_info) != 0:
        logging.error("fail to notify zookeeper")
        return 1

    logging.info("deploy %s succ", args.date)

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
