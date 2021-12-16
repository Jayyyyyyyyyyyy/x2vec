#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import urllib2


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
    url = "http://10.42.43.101:8088/search?num={}&key={}".format(num, vid)
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

    while True:
        vid = raw_input("please input a vid: ")
        if not vid:
            break
        title = get_title(vid)
        print "===== recalls ====="
        print "query title: ", title
        i = 1
        for x, score in get_recalls(vid, 50):
            print i, x, score, get_title(x)
            i += 1
