#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import os
import logging
import argparse
import ConfigParser
import math
import common.datetime_wrapper as datetime
import common.hadoop_shell_wrapper as hadoop_shell
import common.fs_wrapper as fs
import common.shell_wrapper as shell
import common.spark_submit_wrapper as spark
from common.zookeeper_utils import *
from collections import namedtuple
import time
from gensim.models import word2vec, Word2Vec

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
    zk_ack_node = config.get("zookeeper", "ack_node")

    # 连接zookeeper
    zk = zk_connect(zk_host)
    if zk is None:
        logging.error("fail to connect zookeeper %s", zk_host)
        return -1, None

    # 获取在线服务一致使用的redis
    server_redis_list = []

    server_ack_nodes = zk_get_children(zk, zk_ack_node)
    logging.info("servers: %s", server_ack_nodes)
    if server_ack_nodes is not None:
        for server_ack_node in server_ack_nodes:
            server_node_path = "{}/{}".format(zk_ack_node, server_ack_node)
            server_model_str = zk_get(zk, server_node_path)
            server_model_info = parse_model_info(server_model_str)
            if server_model_info is None:
                logging.error("fail to parse server [%s] ack info [%s]", server_node_path, server_model_str)
                return -1, None
            logging.info("server=%s, model hdfs=%s, redis=%s",
                         server_node_path, server_model_info.hdfs, server_model_info.redis)
            server_redis_list.append(server_model_info.redis)

    server_redis_list = list(set(server_redis_list))
    if len(server_redis_list) == 0:
        return 0, None
    if len(server_redis_list) > 1:
        logging.error("all server loaded models are not synchronized")
        return -1, None
    return 0, server_redis_list[0]


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


def fetch_idle_redis(config, args):
    """
    根据zookeeper中当前servers统一的配置，获取当前闲置的redis
    :param config: 配置文件对象
    :param args: 命令行参数
    :return: 闲置redis host（失败：None）
    """
    redis_choices = config.get("feed_redis", "choices").split(";")
    if len(redis_choices) != 2:
        logging.error("feed_redis.choices size must equals 2")
        return -1, None

    ret, active_redis = fetch_synchronized_redis(config, args)
    if ret != 0:
        logging.error("fail to fetch synchronized redis")
        return -1, None

    # 如果server存在，并且使用中redis一致
    if active_redis:
        if active_redis not in redis_choices:
            logging.error("error! servers currently connected redis %s are not among choices %s",
                          active_redis, redis_choices)
            return -1, None
        redis_choices.remove(active_redis)
    idle_redis = redis_choices[0]
    assert idle_redis, "idle redis is empty"
    return 0, idle_redis


def deploy_redis(config, args):
    ret, idle_redis = fetch_idle_redis(config, args)
    if ret != 0 or not idle_redis:
        logging.error("fail to get idle redis host, exit")
        return 1, None

    hdfs_model_path = config.get("common", "hdfs_output", 0,
                                 {'date': args.date, 'dir': config.get("model", "hdfs_dir")})

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
        .set_app_jar(FILE_DIR + "/../lib/" + config.get("common", "jar")) \
        .add_app_argument("input_path", "{}/{}".format(hdfs_model_path, "item.emb")) \
        .add_app_argument("redis", idle_redis) \
        .add_app_argument("version", args.date) \
        .add_app_argument("dim", config.getint("word2vec", "size")) \
        .add_app_argument("concurrency", config.getint("feed_redis", "concurrency")) \
        .add_app_argument("flushall", config.get("feed_redis", "flushall"))

    if ss.run(print_cmd=True, print_info=True):
        logging.info("successfully feed user emb into redis")
        return 0, idle_redis
    logging.error("fail to feed user emb into redis")
    return 1, None


def filter_recommendable(input_emb_file, input_recommendable_file, output_emb_file):
    key_set = set()
    with open(input_recommendable_file, "r") as fin_key:
        for line in fin_key:
            items = line.strip().split('\t')
            if len(items) == 0:
                continue
            if items[0]:
                key_set.add(items[0])
    logging.info("video recommendable filter set size = %d", len(key_set))
    if len(key_set) < 10000:
        logging.error("too few keys in %s", input_recommendable_file)
        return 1
    cnt = 0
    with open(output_emb_file, "w") as fout:
        with open(input_emb_file, "r") as fin:
            for line in fin:
                k, v = line.strip().split('\t')
                if k in key_set:
                    cnt += 1
                    print >> fout, "{}\t{}".format(k, v)
    logging.info("after filtering, %d keys are kept", cnt)
    if cnt < 10000:
        logging.error("after filtering, too few keys are left")
        return 1
    return 0


def generate_index(config, args):
    local_train_path = "{}/../{}".format(
        FILE_DIR,
        config.get(
            "model", "local_dir", 0,
            {
                "date": args.date,
                "project": config.get("common", "project")
            }
        )
    )
    local_item_model_file = os.path.join(local_train_path, "item.emb")
    local_online_path = os.path.join(local_train_path, "i2i-online")
    local_item_model_filtered_file = os.path.join(local_train_path, "item.emb.filter")

    # =============  开始：对检索向量建faiss索引
    if not fs.remove_dir(local_online_path) or not fs.mkdir(local_online_path):
        logging.error("fail to clean online i2i model path %s", local_online_path)
        return 1

    # ========== 开始：过滤待检索集合
    local_recommendable_video_file = os.path.join(local_train_path, "recommendable_videos.txt")
    hdfs_recommendable_video_path = config.get("recommendable_videos", "path", 0, {"date": args.date})
    if not fs.remove_file(local_recommendable_video_file) or \
            not hadoop_shell.exists_all(hdfs_recommendable_video_path, flag=True, print_missing=True) or \
            not hadoop_shell.getmerge(hdfs_recommendable_video_path, local_recommendable_video_file):
        logging.error("fail to download %s", hdfs_recommendable_video_path)
        return 1

    if filter_recommendable(local_item_model_file, local_recommendable_video_file, local_item_model_filtered_file) != 0:
        logging.error("fail to filtering item embedding")
        return 1

    cmd = "{}/../bin/build_faiss_index " \
          "--clusters {} " \
          "--dim {} " \
          "--metric {} " \
          "--samples {} " \
          "--inputs {} " \
          "--output {} " \
          "--version {}".format(FILE_DIR,
                                config.getint("faiss", "clusters"),
                                config.getint("word2vec", "size"),
                                config.get("faiss", "metric"),
                                config.getint("faiss", "samples"),
                                local_item_model_filtered_file,
                                local_online_path,
                                args.date)

    logging.info("build_index cmd: %s", cmd)
    if not shell.shell_command(cmd=cmd, print_info=True):
        logging.error("build_index %s failed", args.date)
        return 1
    logging.info("successfully build_index %s", args.date)
    # =============  结束：对检索向量建faiss索引
    return 0


def notify_zookeeper(config, args, model_info):
    if not args.notify:
        logging.info("notify switch closed, skip notifying zookeeper")
        return 0

    if not isinstance(model_info, ModelInfo) or \
            len(model_info.redis) == 0 or \
            len(model_info.hdfs) == 0:
        logging.error("error model_info, exit notify_zookeeper")
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
        lifetime = config.getint("model", "hdfs_lifetime")
        if lifetime > 0:
            dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
            expire_path = config.get(
                "common", "hdfs_output", 0,
                {'date': dt_expire, 'dir': config.get("model", "hdfs_dir")})
            if hadoop_shell.rmr(expire_path):
                logging.info("successfully removed hdfs expire %s", expire_path)
            else:
                logging.warning("failed to remove hdfs expire %s", expire_path)
                return 1

        lifetime = config.getint("model", "local_lifetime")
        if lifetime > 0:
            dt_expire = datetime.DateTime(args.date).apply_offset_by_day(-lifetime)
            expire_path = "{}/../{}".format(
                FILE_DIR,
                config.get(
                    "model", "local_dir", 0, {"date": dt_expire, "project": config.get("common", "project")}))
            if fs.remove_dir(expire_path):
                logging.info("successfully removed local expire %s", expire_path)
            else:
                logging.warning("failed to remove local expire %s", expire_path)
                return 1
    else:
        logging.info("expire switch close, skip removing expired data")

    return 0


def normailze(vec):
    square_vec = map(lambda x: x ** 2, vec)
    total = math.sqrt(sum(square_vec))
    return map(lambda x: x / total, vec)


def export_word2vec_embedding(input_model_file, output_emb_file, need_normalize):
    try:
        model = Word2Vec.load(input_model_file)
        with open(output_emb_file, "w") as fout_emb:
            for key in model.wv.vocab.iterkeys():
                key_uft8 = key.encode("utf-8", "ignore")
                vec = normailze(model[key]) if need_normalize else model[key]
                fout_emb.write("{}\t{}\n".format(
                    key_uft8,
                    ",".join(["{:.6f}".format(x) for x in vec])
                ))
        return 0
    except Exception as e:
        logging.error(e.message)
    return 1


def run(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    local_train_path = "{}/../{}".format(
        FILE_DIR,
        config.get(
            "model", "local_dir", 0,
            {
                "date": args.date,
                "project": config.get("common", "project")
            }
        )
    )

    hdfs_model_path = config.get("common", "hdfs_output", 0,
                                 {'date': args.date, 'dir': config.get("model", "hdfs_dir")})

    local_model_path = os.path.join(local_train_path, "model")
    local_model_file = os.path.join(local_model_path, "word2vec.model")
    local_item_emb_file = os.path.join(local_train_path, "item.emb")
    local_online_path = os.path.join(local_train_path, "i2i-online")
    hdfs_online_model = "{}/{}".format(hdfs_model_path, "i2i-online")
    local_train_succ_file = os.path.join(local_train_path, "_TRAIN_SUCCESS")
    local_deploy_succ_file = os.path.join(local_train_path, "_DEPLOY_SUCCESS")

    if fs.exists(local_deploy_succ_file):
        logging.info("deploy succ flag found! skip")
        return 0

    # 判断向量文件是否存在
    if not fs.exists(local_train_succ_file) or not fs.exists(local_model_file):
        logging.error("model not found complete, exit")
        return 1

    if export_word2vec_embedding(
            local_model_file,
            local_item_emb_file,
            config.getboolean("model", "vector_normalization")) != 0:
        logging.error("fail to export embedding from word2vec model")
        return 1

    # 清理hdfs的模型目录
    if not hadoop_shell.rmr(hdfs_model_path) or not hadoop_shell.mkdir(hdfs_model_path):
        logging.error("fail to clear hdfs model path: %s", hdfs_model_path)
        return 1

    logging.info("===== start to upload user & item embedding to hdfs =====")
    # 上传用户/视频向量文件
    if not hadoop_shell.put(local_item_emb_file, hdfs_model_path):
        logging.error("fail to put embedding to hdfs %s", hdfs_model_path)
        return 1

    logging.info("===== start to feed item query embedding into redis =====")
    # 把向量灌入备份redis
    ret, idle_redis = deploy_redis(config, args)
    if ret != 0 or not idle_redis:
        logging.error("fail to import user embedding to redis")
        return 1

    logging.info("===== start to generate faiss index =====")
    # 生成视频向量的faiss索引
    if generate_index(config, args) != 0:
        logging.error("failed generate_index")
        return 1

    logging.info("===== start to upload faiss index =====")
    # 把模型上传至hdfs，server会从hdfs拉取模型
    if not hadoop_shell.put(local_online_path, hdfs_model_path):
        logging.error("fail to put model to hdfs %s", hdfs_model_path)
        return 1

    # 通知zookeeper更新模型
    model_info = ModelInfo(hdfs=hdfs_online_model, redis=idle_redis)
    if notify_zookeeper(config, args, model_info) != 0:
        logging.error("fail to notify zookeeper")
        return 1

    logging.info("deploy %s succ", args.date)

    logging.info("===== start to remove expire data =====")
    # 删除过期本地模型目录和hdfs目录
    if remove_expire(config, args) != 0:
        logging.error("fail to remove expire")
        return 1

    logging.info("===== start to backup word2vec raw model =====")
    if not hadoop_shell.put(local_model_path, hdfs_model_path):
        logging.error("fail to upload dnn-model")
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
