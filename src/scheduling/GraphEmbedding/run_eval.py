#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

import os
import logging
import argparse
from gensim.models import word2vec, Word2Vec
from sklearn import metrics


FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def auc_score(positives, negatives):
    labels = [1] * len(positives)
    scores = positives
    labels.extend([0] * len(negatives))
    scores.extend(negatives)
    auc = metrics.roc_auc_score(labels, scores)
    return auc


def run(args):
    model = Word2Vec.load(args.model)
    auc_total = 0
    auc_cnt = 0
    with open(args.sample, "r") as fin:
        for line in fin:
            cols = line.strip().split('\t')
            if len(cols) != 3:
                continue
            pivot_vid = cols[0].decode("utf-8", "ignore")
            positive_vid = cols[1].decode("utf-8", "ignore")
            negative_vids = map(lambda v: v.decode("utf-8", "ignore"), cols[2].split(' '))
            if pivot_vid not in model.wv.vocab or positive_vid not in model.wv.vocab:
                continue
            positive_score = model.wv.similarity(pivot_vid, positive_vid)
            negative_scores = [
                model.wv.similarity(pivot_vid, x) for x in negative_vids if x in model.wv.vocab
            ]

            if len(negative_scores) < 3:
                continue
            auc = auc_score([positive_score], negative_scores)
            auc_total += auc
            if auc_cnt % 10000 == 0:
                logging.info("eval process: %d", auc_cnt)
            auc_cnt += 1
            if 0 < args.num <= auc_cnt:
                break
    logging.info("auc=%f (%f/%d)", auc_total / auc_cnt, auc_total, auc_cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train bpr embedding')
    parser.add_argument('--model', dest='model', required=True, help="model path")
    parser.add_argument('--sample', dest='sample', required=True, help="sample path")
    parser.add_argument('--num', dest='num', required=False, default=0, type=int, help="sample num")
    arguments = parser.parse_args()

    try:
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
