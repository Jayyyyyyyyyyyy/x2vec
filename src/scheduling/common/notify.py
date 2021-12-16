# -*- coding: utf-8 -*-

import sys
import json
import os
import urllib2

reload(sys)
sys.path.append(os.getcwd())


def notify_sms(phones, msg):
    if isinstance(phones, list):
        phones = ",".join(phones)
    url = "http://10.10.77.161/sms.php"
    content = "tos={}&content={}".format(phones, msg)
    try:
        req = urllib2.Request(url=url, data=content)
        urllib2.urlopen(req)
    except Exception as e:
        print >> sys.stderr, e


def notify_im(msg, **args):
    content = msg
    for k, v in args.viewitems():
        content += "\n > {}：<font color=\"warning\">{}</font>".format(k, v)

    text_dict = {
        "msgtype": "markdown",
        "markdown": {
            "content": content
        }
    }
    text_json = json.dumps(text_dict)
    header_dict = {"Content-Type": "application/json"}
    url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=ac428ee8-c6b6-48d7-ae61-649f8a9f6ea9"
    try:
        req = urllib2.Request(url=url, data=text_json, headers=header_dict)
        urllib2.urlopen(req)
    except Exception as e:
        print >> sys.stderr, e
