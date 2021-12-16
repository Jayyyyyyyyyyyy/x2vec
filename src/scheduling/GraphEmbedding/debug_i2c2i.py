#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import urllib2
import argparse


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


def get_i2c(diu, history_len, cluster_variance, rank_time_decay, leader_alg):
    url = "http://10.42.12.90:9077/vector/clusterDebug?diu={}&history={}&variance={}&timeDecay={}&leader={}".format(
        diu, history_len, cluster_variance, rank_time_decay, leader_alg)
    try:
        req = urllib2.Request(url=url)
        res = urllib2.urlopen(req)
        res = res.read()
        res = json.loads(res)
        return res
    except Exception as e:
        pass
    return {}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='build samples')
    parser.add_argument('--diu', dest='diu', required=True)
    parser.add_argument('--history', dest='history', type=int, required=False, default=30)
    parser.add_argument('--variance', dest='variance', type=float, required=False, default=1.5)
    parser.add_argument('--time_decay', dest='time_decay', type=float, required=False, default=0.06)
    parser.add_argument('--leader', dest='leader', type=str, required=False, default="pivot")
    arguments = parser.parse_args()

    i2c_result = get_i2c(arguments.diu, arguments.history, arguments.variance, arguments.time_decay, arguments.leader)
    if "clusters" not in i2c_result or len(i2c_result["clusters"]) == 0:
        print "empty clusters!"
        sys.exit(0)

    history = i2c_result["history"]
    title_map = {}

    print "========== history size = {}".format(len(history))
    for history_idx, vid in enumerate(history):
        title = get_title(vid)
        title_map[vid] = title
        print "{}. {}  {}".format(history_idx, vid, title)

    for cluster_idx, cluster in enumerate(i2c_result["clusters"]):
        cluster_leader = cluster["pivot"]
        print "========== cluster: {}, size = {}".format(cluster_idx, cluster["size"])
        for member_idx, vid in enumerate(cluster["member"]):
            print "{}. leader:{}, idx {} of history, {}  {}".format(member_idx, 1 if cluster_leader == vid else 0, history.index(vid), vid, title_map[vid])
