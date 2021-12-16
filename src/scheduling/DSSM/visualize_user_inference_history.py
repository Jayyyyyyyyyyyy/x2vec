#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import urllib2


def get_video_profile(vid):

    url = "http://10.10.101.226:9221/portrait_video/video/{}".format(vid)
    try:
        req = urllib2.Request(url=url)
        res = urllib2.urlopen(req)
        res = res.read()
        res = json.loads(res)
        if res["found"]:
            return res["_source"]
    except Exception as e:
        pass
    return {}


if __name__ == '__main__':
    vocab = {}
    with open("vid.vocab", "r") as fin:
        for line in fin:
            cols = line.strip().split("\t")
            vocab[cols[0]] = cols[1]

    with open("inference_sample.txt", "r") as fin:
        for line in fin:
            raw_input("enter to proceed...")
            cols = line.strip().split("\t")
            if len(cols) != 3:
                continue
            diu = cols[0]
            history = cols[2].split(",")

            i = 1
            for modelid in history:

                vid = vocab[modelid]
                profile = get_video_profile(vid)
                title = profile.get("title", u"null").encode("utf-8", "ignore")
                content_genre_dict = profile.get("content_genre", {})
                content_genre = content_genre_dict.get("tagname", u"null").encode("utf-8", "ignore") if isinstance(content_genre_dict, dict) else "null"
                firstcat_dict = profile.get("firstcat", {})
                firstcat = firstcat_dict.get("tagname", u"null").encode("utf-8", "ignore") if isinstance(firstcat_dict, dict) else "null"
                secondcat_dict = profile.get("secondcat", {})
                secondcat = secondcat_dict.get("tagname", u"null").encode("utf-8", "ignore") if isinstance(secondcat_dict, dict) else "null"
                content_tag = []
                for x in profile.get("content_tag", {}):
                    if isinstance(x, dict):
                        content_tag.append(x.get("tagname", u"null").encode("utf-8", "ignore"))
                mp3_dict = profile.get("content_mp3", {})
                mp3 = mp3_dict.get("tagname", u"null").encode("utf-8", "ignore") if isinstance(mp3_dict, dict) else "null"
                uname = profile.get("uname", u"null").encode("utf-8", "ignore")
                print >> sys.stdout, i, vid, title, "|", content_genre, "|", firstcat, "|", secondcat, "|", "|".join(content_tag), "|", mp3, "|", uname
                i += 1


