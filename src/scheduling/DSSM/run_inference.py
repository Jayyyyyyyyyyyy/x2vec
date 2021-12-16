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
import common.fs_wrapper as fs
from model_configuration import ModelConfiguration
from dssm import DssmModel

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def convert_embedding_id(id_emb_file, vocab_file, output_emb_file):
    id_dict = {}
    with open(vocab_file, "r") as fin:
        for line in fin:
            cols = line.strip().split("\t")[:2]
            id_dict[cols[0]] = cols[1]
    logging.info("load %d mapping pairs from %s", len(id_dict), vocab_file)
    if len(id_dict) < 1000:
        logging.error("too few id, failed")
        return 1

    keep_cnt = 0
    with open(output_emb_file, "w") as fout:
        with open(id_emb_file, "r") as fin:
            for line in fin:
                cols = line.strip().split("\t")
                if len(cols) != 2:
                    continue
                if cols[0] not in id_dict:
                    continue
                keep_cnt += 1
                fout.write("{}\t{}\n".format(id_dict[cols[0]], cols[1]))
    logging.info("after converting, %d emb are kept", keep_cnt)
    if keep_cnt < 1000:
        logging.error("too few left")
        return 1
    return 0


def run(args):
    config = ConfigParser.SafeConfigParser()
    config.read(args.conf)

    local_train_path = "{}/../{}".format(
        FILE_DIR,
        config.get("model", "local_dir", 0,
                   {
                       "date": args.date,
                       "project": config.get("common", "project")
                   }))
    local_model_path = os.path.join(local_train_path, "model")
    local_user_embedding = os.path.join(local_train_path, "user.emb")
    local_item_embedding = os.path.join(local_train_path, "item.emb")
    local_item_embedding_raw = local_item_embedding + ".raw"
    local_vid_vocab_file = os.path.join(local_train_path, "vid.vocab")

    local_inference_succ_flag = os.path.join(local_train_path, "_INFERENCE_SUCCESS")
    if fs.exists(local_inference_succ_flag):
        logging.info("inference succ flag found, skip")
        return 0

    hdfs_samples_path = config.get("common", "hdfs_output", 0, {
        "date": args.date,
        "dir": config.get("inference_sample", "tfrecord_dir")
    })

    # 检查训练样本是否存在集群上
    if not hadoop_shell.exists_all(
            hdfs_samples_path,
            flag=True,
            print_missing=True):
        logging.error("train samples missing on hdfs")
        return 1

    # 加载训练配置
    model_conf_file = os.path.join(local_train_path, "model.conf")
    if not fs.exists(model_conf_file):
        logging.error("model training conf file not found")
        return 1
    model_conf = ModelConfiguration().load(model_conf_file)
    if not model_conf.check():
        logging.error("model conf invalid!")
        return 1

    yarn_masters = eval(config.get("yarn_rec", "masters"))
    valid_master = hadoop_shell.first_exist(*yarn_masters, print_cmd=True, flag=False)
    if not valid_master:
        logging.error("no yarn master available")
        return 1
    modules_and_feeds = {"inference": valid_master + hdfs_samples_path}

    # 训练模型
    model = DssmModel(model_conf, local_model_path, **modules_and_feeds)
    if model.inference(local_user_embedding) != 0:
        logging.error("inference failed")
        return 1

    if model.export_embedding(local_item_embedding_raw) != 0:
        logging.error("export embedding failed")
        return 1

    if not fs.exists(local_vid_vocab_file):
        hdfs_vid_vocab = config.get("common", "hdfs_output", 0, {
            "date": args.date,
            "dir": config.get("vocab", "vid_dir")
        })
        if not hadoop_shell.exists_all(hdfs_vid_vocab, flag=True, print_missing=True):
            logging.error("vocab vid not found on hdfs")
            return 1
        if not hadoop_shell.getmerge(hdfs_vid_vocab, local_vid_vocab_file) or \
                not fs.exists(local_vid_vocab_file):
            logging.error("fail to download vocab vid")
            return 1

    if convert_embedding_id(local_item_embedding_raw, local_vid_vocab_file, local_item_embedding) != 0:
        logging.error("fail to convert id")
        return 1

    logging.info("inference succ")
    fs.touch(local_inference_succ_flag)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train bpr embedding')
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
