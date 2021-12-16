#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import os
import json
import logging
import argparse
import ConfigParser
from common.notify import *
import common.datetime_wrapper as datetime
from common.notify import *
from common.sys_utils import *
import urllib2

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S')


def is_service_ok(server):
    url = "http://{}/info".format(server)
    try:
        req = urllib2.Request(url=url)
        res = urllib2.urlopen(req)
        res = res.read()
        res_dict = json.loads(res)
        return res_dict["status"] == 0 and res_dict["msg"] == "OK"
    except Exception as e:
        pass
    return False


def run(args):

    with open(args.conf, "r") as fin:
        service_conf = json.loads(fin.read())
        logging.info("%s", json.dumps(service_conf, ensure_ascii=False, indent=4))

        for service_name, servers in service_conf.viewitems():
            for server in servers:
                if not is_service_ok(server):
                    msg = "x2vec-ann-angine crashed! service={}, host={}".format(
                        service_name, server
                    )
                    notify_sms("18510411406", msg)
                    notify_im("x2vec-ann-angine crashed", service=service_name, server=server, owner="wangshuang")
                    logging.error("service is bad: %s @ %s", service_name, server)
                else:
                    logging.info("service is ok: %s @ %s", service_name, server)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='monitor')
    parser.add_argument('--conf', dest='conf', required=True, help="conf file")
    arguments = parser.parse_args()

    try:
        sys.exit(run(arguments))
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        sys.exit(1)
