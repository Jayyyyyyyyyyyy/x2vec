# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import common.fs_wrapper as fs
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import argparse
import urllib2
import json


def get_title(vid):

    url = "http://10.10.101.226:9221/portrait_video/video/{}".format(vid)
    try:
        req = urllib2.Request(url=url)
        res = urllib2.urlopen(req)
        res = res.read()
        res = json.loads(res)
        if res["found"]:
            return res["_source"].get("title", u"").encode("utf-8", "ignore")
    except Exception as e:
        pass
    return ""


class Visualizer(object):
    def __init__(self, emb_file, num, output_path):
        self.emb_file = emb_file
        self.output_path = output_path

        self.embs = []
        self.meta = []
        self.dimension = -1

        fs.mkdir(output_path)

        cnt = 0
        with open(self.emb_file, 'r') as fin:
            for i, line in enumerate(fin):
                if cnt >= num > 0:
                    break
                cols = line.strip().split("\t")
                if len(cols) != 2:
                    continue
                title = get_title(cols[0])
                if title and "\n" not in title and "\r" not in title:
                    self.embs.append(cols[1])
                    self.meta.append(title)
                    if self.dimension < 0:
                        self.dimension = len(cols[1].split(","))
                    cnt += 1

    def visualize(self):

        with open(self.output_path + "/meta.txt", "w") as fout:
            for title in self.meta:
                fout.write(title + "\n")

        # adding into projector
        config = projector.ProjectorConfig()

        placeholder = np.zeros((len(self.embs), self.dimension))

        for i, line in enumerate(self.embs):
            placeholder[i] = np.fromstring(line, sep=',')

        embedding_var = tf.Variable(placeholder, trainable=False, name='emb')

        embed = config.embeddings.add()
        embed.tensor_name = embedding_var.name
        embed.metadata_path = "meta.txt"

        # define the model without training
        sess = tf.InteractiveSession()

        tf.global_variables_initializer().run()
        saver = tf.train.Saver()

        saver.save(sess, os.path.join(self.output_path, 'emb_metadata.ckpt'))

        writer = tf.summary.FileWriter(self.output_path, sess.graph)
        projector.visualize_embeddings(writer, config)
        sess.close()
        print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(self.output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate file for visualizing embeddings in tensorboard')
    parser.add_argument('--emb_file', dest='emb_file', required=True, help="emb_file")
    parser.add_argument('--output', dest='output', required=True, help="model path")
    parser.add_argument('--num', dest='num', required=True, type=int, help="num")
    arguments = parser.parse_args()

    visualizer = Visualizer(arguments.emb_file, arguments.num, arguments.output)
    visualizer.visualize()
