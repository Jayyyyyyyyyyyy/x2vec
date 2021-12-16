#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
import numpy as np
from annoy import AnnoyIndex
import argparse
import ConfigParser
import os
import logging
import common.hadoop_shell_wrapper as hadoop_shell
import common.fs_wrapper as fs

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def do_knn(query_filename, index_filename, output_filename, n_trees, dim, top_n, score_threshold):

    index = AnnoyIndex(dim, 'dot')
    index_vid_list = []

    with open(index_filename, "r") as fin:
        idx = 0
        for line in fin:
            cols = line.split("\t", 2)
            if len(cols) != 2:
                continue
            vid = cols[0]
            vector = map(float, cols[1].split(","))
            if len(vector) != dim:
                logging.error("vector size not match, expected=%d, real=%d", dim, len(vector))
                continue
            index.add_item(idx, vector)
            index_vid_list.append(vid)
            idx += 1

    logging.info("all index vectors loaded from %s", index_filename)

    index.build(n_trees)

    logging.info("build index successfully!")

    cnt = 0
    with open(query_filename, "r") as fin:

        with open(output_filename, "w") as fout:
            for line in fin:
                cols = line.split("\t", 2)
                if len(cols) != 2:
                    continue
                vid = cols[0]
                vector = map(float, cols[1].split(","))
                if len(vector) != dim:
                    logging.error("vector size not match, expected=%d, real=%d", dim, len(vector))
                    continue
                knn_idx_list, knn_score_list = index.get_nns_by_vector(vector, top_n + 1, search_k=-1, include_distances=True)
                knn_result = filter(lambda x: x[0] != vid and x[1] >= score_threshold, map(lambda x: (index_vid_list[x[0]], x[1]), zip(knn_idx_list, knn_score_list)))[:top_n]

                if len(knn_result) > 0:
                    fout.write("{}\t{}\n".format(vid, " ".join(map(lambda x: "{}:{}".format(x[0], x[1]), knn_result))))
                    cnt += 1
                    if cnt % 5000 == 0:
                        logging.info("%d query processed", cnt)

    logging.info("totally %d query result saved!", cnt)
    return 0 if cnt > 0 else 1


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
    hdfs_model_path = config.get("common", "hdfs_output", 0, {'date': args.date, 'dir': config.get("model", "hdfs_dir")})
    local_deploy_succ_file = os.path.join(local_train_path, "_DEPLOY_SUCCESS")
    local_query_emb_file = os.path.join(local_train_path, "item.emb")
    local_index_emb_file = os.path.join(local_train_path, "item.emb.filter")
    local_output_knn_file = os.path.join(local_train_path, "knn.out")

    if not fs.exists(local_deploy_succ_file):
        logging.error("deploy succ flag not found! model not ready!")
        return 1

    if do_knn(
            local_query_emb_file,
            local_index_emb_file,
            local_output_knn_file,
            n_trees=100,
            dim=config.getint("word2vec", "size"),
            top_n=30,
            score_threshold=0.6) != 0:
        logging.error("fail to run knn search!")
        return 1

    if not hadoop_shell.put(local_output_knn_file, hdfs_model_path):
        logging.error("fail to put knn result file to hdfs")
        return 1
    logging.info("successfully put knn result to hdfs")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='deploy and backup model')
    parser.add_argument('--date', dest='date', required=True, help="which date(%%Y-%%m-%%d)")
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    args = parser.parse_args()

    sys.exit(run(args))
