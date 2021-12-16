# -*- coding: utf-8 -*-

import sys
import os
import time
import logging
import hadoop_shell_wrapper as hadoop_shell

reload(sys)
sys.path.append(os.getcwd())


class HdfsLock(object):
    def __init__(self, lock_file, attempt_interval, timeout):
        self._lock_file = lock_file
        self._attempt_interval = max(attempt_interval, 1)  # sec
        self._timeout = timeout  # sec
        self._acquired = False

    def acquire(self):
        if self._acquired:
            logging.warning("hdfs lock already occupied, fail to occupy twice")
            return False
        self._acquired = False
        begin = time.time()
        while time.time() - begin < self._timeout:
            if hadoop_shell.exists_all(self._lock_file):
                logging.warning("hdfs lock occupied by others, wait...")
            else:
                if hadoop_shell.touchz(self._lock_file):
                    logging.info("hdfs lock acquired")
                    self._acquired = True
                    return True
                else:
                    logging.info("hdfs lock free, but fail to acquire, wait...")
            time.sleep(self._attempt_interval)
        return False

    def release(self):
        if self._acquired:
            self._acquired = not hadoop_shell.rm(self._lock_file)
            return not self._acquired
        return True


# python hdfs_lock.py
if __name__ == "__main__":
    logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    lock = HdfsLock("/user/hadoop/temp_file_lock", 2, 10)
    print lock.acquire()
    print lock.acquire()
    print lock.release()
