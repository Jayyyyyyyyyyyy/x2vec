# -*- coding: utf-8 -*-

import sys
import os
import zookeeper
import time
reload(sys)
sys.path.append(os.getcwd())


def zk_connect(hosts, retry=10, interval=2):
    zk = zookeeper.init(hosts)
    attempt = 1
    while attempt <= retry:
        time.sleep(interval)
        try:
            zookeeper.get_children(zk, "/")
            return zk
        except Exception as ex:
            print >> sys.stderr, ex
        attempt += 1
    return None


def zk_exists(zk, path):
    try:
        return zookeeper.exists(zk, path) is not None
    except Exception as ex:
        print >> sys.stderr, ex
    return False


def zk_close(zk):
    try:
        if zk is not None:
            zookeeper.close(zk)
    except Exception as ex:
        print >> sys.stderr, ex


def zk_create(zk, path):
    if path.endswith("/"):
        print >> sys.stderr, "zk path must not end with slash '/'"
        return False
    idx = path.find("/", 1)
    try:
        while idx > 0:
            subpath = path[:idx]
            if not zk_exists(zk, subpath):
                zookeeper.create(zk, subpath, "", [{"perms":"crdwa", "scheme":"world", "id":"anyone"}], 0)
            idx = path.find("/", idx + 1)
        if not zk_exists(zk, path):
            zookeeper.create(zk, path, "", [{"perms":"crdwa", "scheme":"world", "id":"anyone"}], 0)
    except Exception as ex:
        print >> sys.stderr, ex
        return False
    return zk_exists(zk, path)


def zk_set(zk, path, value):
    try:
        zookeeper.set(zk, path, value)
    except Exception as ex:
        print >> sys.stderr, ex
        return False
    return True


def zk_get(zk, path):
    try:
        res = zookeeper.get(zk, path)
        if (isinstance(res, tuple) or isinstance(res, list)) and len(res) == 2 and isinstance(res[0], str):
            return res[0]
    except Exception as ex:
        print >> sys.stderr, ex
    return None


def zk_get_children(zk, path):
    try:
        res = zookeeper.get_children(zk, path, None)
        if isinstance(res, list) or isinstance(res, tuple):
            return res
    except Exception as ex:
        print >> sys.stderr, ex
    return None


# python zookeeper_utils.py
if __name__ == "__main__":
    pass
