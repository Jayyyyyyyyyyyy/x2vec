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
    url = "http://10.42.43.101:8086/search?num={}&key={}".format(num, vid)
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

def get_itemcf(vid, num):
    url = "http://10.42.28.45:8080/recall/itemcf?id={}&num={}".format(vid, num)
    try:
        req = urllib2.Request(url=url)
        res = urllib2.urlopen(req)
        res = res.read()
        res = json.loads(res)
        return [x["id"].encode("utf-8", "ignore") for x in res["data"]]
    except Exception as e:
        pass
    return []

if __name__ == '__main__':

    same_cnt = 0
    all_cnt = 0
    with open("videos", "r") as fin:
        i = 1
        for line in fin:
            vid = line.strip()
            if not vid:
                continue
            itemcfs = set(get_itemcf(vid, 100))
            model_sims = get_recalls(vid, 100)
            same = len([x for x in model_sims if x[0] in itemcfs])
            same_cnt += same
            all_cnt += 10
            i += 1
            if i >= 10000:
                break
    print same_cnt, all_cnt, same_cnt*1.0/all_cnt
