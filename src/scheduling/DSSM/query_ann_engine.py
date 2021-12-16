#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import urllib2
from gensim.models import word2vec, Word2Vec


def get_title(vid):

    url = "http://10.10.101.226:9221/portrait_video/video/{}".format(vid)
    try:
        req = urllib2.Request(url=url)
        res = urllib2.urlopen(req)
        res = res.read()
        res = json.loads(res)
        if res["found"]:
            return res["_source"].get("title", u"null").encode("utf-8", "ignore")
    except Exception as e:
        pass
    return "null"


def get_recalls(vid, num):
    url = "http://10.42.43.101:8085/search?num={}&key={}".format(num, vid)
    try:
        req = urllib2.Request(url=url)
        res = urllib2.urlopen(req)
        res = res.read()
        res = json.loads(res)
        #return [[x["id"].encode("utf-8", "ignore"), x["score"], x["rank_score"]] for x in res["data"]]
        return [[x["id"].encode("utf-8", "ignore"), x["score"]] for x in res["data"]]
    except Exception as e:
        pass
    return []


if __name__ == '__main__':
    vocab = {}
    with open("vid.vocab", "r") as fin:
        for line in fin:
            cols = line.strip().split("\t")
            vocab[cols[0]] = cols[1]

    historys = {}
    with open("user_feeds.txt", "r") as fin:
        for line in fin:
            cols = line.strip().split("\t")
            historys[cols[0]] = cols[2].split(",")

    while True:
        vid = raw_input("please input a diu: ")
        if not vid:
            break
        print "==== history ======"
        if vid in historys:
            history = historys[vid]
            i = 1
            for v in history:
                print i, vocab[v], get_title(vocab[v])
                i += 1

        print "===== recalls ====="
        i = 1
        for x, score in get_recalls(vid, 30):
            print i, x, score, get_title(x)
            i += 1
