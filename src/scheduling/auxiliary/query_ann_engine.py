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

    """
    video_stats = {}
    with open("video_stats.txt", "r") as fin:
        for line in fin:
            cols = line.strip().split("\t")
            if len(cols) != 7:
                continue
            video_stats[cols[0]] = [float(cols[1]), float(cols[2]), float(cols[3]), float(cols[4])]
    """
    #model = Word2Vec.load("../model/GraphEmbedding/2019-08-08/model/word2vec.model")
    while True:
        vid = raw_input("please input a vid: ")
        if not vid:
            break
        title = get_title(vid)
        print "===== recalls ====="
        print "query title: ", title
        i = 1
        itemcfs = get_itemcf(vid, 30)
        #model_sims = model.most_similar(vid.decode("utf-8", "ignore"), topn=30)
        for x, score in get_recalls(vid, 30):
            itemcf_item = itemcfs[i-1] if i <= len(itemcfs) else "null"
            itemcf_title = get_title(itemcf_item)
            #model_item = model_sims[i-1][0].encode("utf-8", "ignore") if i <= len(model_sims) else "null"
            #model_score = model_sims[i-1][1] if i <= len(model_sims) else -1
            #model_title = get_title(model_item)
            #print i, x, score, get_title(x), "|||||",itemcf_item, itemcf_title
            #print i, x, score, get_title(x), "|||||", model_item, model_title, model_score, "|||||", itemcf_item, itemcf_title
            print i, x, score, get_title(x), "|||||", itemcf_item, itemcf_title
            i += 1
