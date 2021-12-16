#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../../")
import os
import logging
import argparse
import math
import json
import time
import ConfigParser
from kafka import KafkaConsumer
import redis
import struct

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def normalize(vec):
    square_vec = map(lambda x: x ** 2, vec)
    total = math.sqrt(sum(square_vec))
    return map(lambda x: x / total, vec)


def run(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    bert_dim = config.getint("bert", "dim")
    should_normalize = config.getboolean("bert", "normalize")

    # ===== connect redis (feed emebdding online)
    redis_host = config.get("feed_redis", "redis_host")
    redis_port = config.getint("feed_redis", "redis_port")
    redis_ttl = config.getint("feed_redis", "ttl")
    logging.info("connecting query redis: host=%s, port=%d, ttl=%d(sec)", redis_host, redis_port, redis_ttl)
    redis_manager = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

    # ===== connect kafka (read video titles)
    kafka_servers = config.get("realtime", "kafka_servers")
    kafka_topic = config.get("realtime", "kafka_topic")
    kafka_groupid = config.get("common", "project")
    logging.info("connecting kafka: servers=%s, topic=%s, groupid=%s", kafka_servers, kafka_topic, kafka_groupid)
    input_kafka_consumer = KafkaConsumer(kafka_topic, group_id=kafka_groupid, bootstrap_servers=kafka_servers)

    for data in input_kafka_consumer:
        try:
            data_dict = json.loads(data.value)
            vid = str(data_dict.get("vid", ""))
            embed = map(float, data_dict.get("embed", "").split(","))
            if len(embed) != bert_dim:
                logging.error("response embedding size %d != %d", len(embed), bert_dim)
                continue
            if should_normalize:
                embed = normalize(embed)

            # write embedding to redis (query end)
            embed_binary = struct.pack("f" * bert_dim, *embed)
            if redis_ttl > 0:
                ret = redis_manager.set(vid, embed_binary, ex=redis_ttl)
            else:
                ret = redis_manager.set(vid, embed_binary)
            if not ret:
                logging.error("fail to feed redis: %s", vid)

            logging.info("vid = %s", vid)

        except Exception as e:
            logging.exception(e.message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bert predict streaming')
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    arguments = parser.parse_args()
    try:
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
