# -*- coding: utf-8 -*-

import sys
import time
import os
import threading
import logging
from collections import namedtuple
import shell_wrapper
reload(sys)
sys.path.append(os.getcwd())

LOG_KEY = "scheduler"
DEFAULT_MAX_THREAD = 2

TASK_STATUS_UNKNOWN = "task_unknown"
TASK_STATUS_READY = "task_ready"
TASK_STATUS_RUNNING = "task_running"
TASK_STATUS_FINISHED = "task_finished"
TASK_STATUS_ERROR = "task_error"
TASK_STATUS_SKIP = "task_skip"


class Task(object):
    def __init__(self, cid, cmd, is_crucial):
        self.cmd = cmd
        self.cid = cid
        self.crucial = is_crucial
        self.dependency_cids = set()
        self.following_cids = set()
        self.status = TASK_STATUS_UNKNOWN


class TaskRecorder(object):
    def __init__(self):
        self.pool = []
        self.mutex = threading.Lock()

    def add(self, cmd, status, start_time, finish_time):
        self.mutex.acquire()
        TaskRecord = namedtuple("TaskRecord", ["cmd", "status", "start_time", "finish_time"])
        tr = TaskRecord(
                cmd=cmd,
                status=status,
                start_time=start_time,
                finish_time=finish_time)
        self.pool.append(tr)
        self.mutex.release()

    def sort(self):
        self.pool.sort(key=lambda x: x.start_time)


class TopologyQueue(object):
    def __init__(self):
        self.tasks = {}
        self.queue = []
        self.mutex = threading.Lock()

    def add_task(self, cid, cmd, is_crucial):
        self.tasks[cid] = Task(cid, cmd, is_crucial)

    def set_dependency(self, dependency_cid, following_cid):
        if dependency_cid not in self.tasks or following_cid not in self.tasks:
            raise Exception("make sure cids [%s , %s] are already defined" % (dependency_cid, following_cid))
        self.tasks[dependency_cid].following_cids.add(following_cid)
        self.tasks[following_cid].dependency_cids.add(dependency_cid)

    def is_cid_acyclic(self, cid):
        ret = False
        for next_cid in self.tasks[cid].following_cids:
            if next_cid == cid or self.is_cid_acyclic(next_cid):
                ret = True
                break
        return ret

    def is_acyclic(self):
        for cid in self.tasks:
            if self.is_cid_acyclic(cid):
                return True
        return False

    def sort(self):
        if self.is_acyclic():
            raise Exception("error!!! acyclic task!!!")
        self.queue = []
        cid_indegree_map = {}
        for cid in self.tasks:
            cid_indegree_map[cid] = len(self.tasks[cid].dependency_cids)
            if cid_indegree_map[cid] == 0:
                self.tasks[cid].status = TASK_STATUS_READY
        while len(cid_indegree_map) > 0:
            del_cnt = 0
            for cid in cid_indegree_map.keys():
                if cid_indegree_map[cid] == 0:
                    self.queue.append(self.tasks[cid])
                    for folowing_cid in self.tasks[cid].following_cids:
                        cid_indegree_map[folowing_cid] -= 1
                    del cid_indegree_map[cid]
                    del_cnt += 1
            if del_cnt == 0:
                raise Exception("topology sort error: %s" % repr(cid_indegree_map))
        if len(self.queue) != len(self.tasks):
            raise Exception("topology sort error: %d - %d" % (len(self.queue), len(self.tasks)))

    def next_task(self):
        ready_task = None
        self.mutex.acquire()
        for cid in self.tasks:
            if self.tasks[cid].status != TASK_STATUS_UNKNOWN:
                continue
            all_finish = True
            for dependency_cid in self.tasks[cid].dependency_cids:
                if self.tasks[dependency_cid].status == TASK_STATUS_FINISHED or \
                        (self.tasks[dependency_cid].status == TASK_STATUS_ERROR and
                         not self.tasks[dependency_cid].crucial):
                    continue
                all_finish = False
                break
            if all_finish:
                self.tasks[cid].status = TASK_STATUS_READY

        for task in self.queue:
            if task.status == TASK_STATUS_READY:
                task.status = TASK_STATUS_RUNNING
                ready_task = task
                break
        self.mutex.release()
        return ready_task

    def __skip_tasks(self, cid):
        travelled = set()
        cids = [cid]
        idx = 0
        while idx < len(cids):
            cur_cid = cids[idx]
            idx += 1
            if cur_cid in travelled:
                continue
            travelled.add(cur_cid)
            for fol_cid in self.tasks[cur_cid].following_cids:
                self.tasks[fol_cid].status = TASK_STATUS_SKIP
                cids.append(fol_cid)

    def finish_task(self, cid, is_succ):
        self.mutex.acquire()
        self.tasks[cid].status = TASK_STATUS_FINISHED if is_succ else TASK_STATUS_ERROR
        if not is_succ and self.tasks[cid].crucial:
            self.__skip_tasks(cid)
        self.mutex.release()

    def terminate(self):
        decision = True
        self.mutex.acquire()
        for task in self.queue:
            if task.status in [TASK_STATUS_UNKNOWN, TASK_STATUS_READY]:
                decision = False
        self.mutex.release()
        return decision

    def size(self):
        self.mutex.acquire()
        queue_len = len(self.queue)
        self.mutex.release()
        return queue_len

    def print_console(self):
        logging.info("-------- task status --------")
        for task in self.queue:
            logging.info("[%s] [%s] [%s]", task.cid, task.cmd, task.status)

    def status(self):
        is_ok = True
        for cid, task in self.tasks.viewitems():
            if task.status != TASK_STATUS_FINISHED and task.crucial:
                is_ok = False
        return is_ok

    def failed_stage(self):
        for cid, task in self.tasks.viewitems():
            if task.status == TASK_STATUS_ERROR and task.crucial:
                return task.cid
        return None


def get_current_timestring():
    tm = time.localtime()
    return "%d-%d-%d %d:%d:%d" % (tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)


class TaskRunner(threading.Thread):
    def __init__(self, tid, task_queue, task_recorder=None):
        threading.Thread.__init__(self)
        self.tid = tid
        self.task_queue = task_queue
        self.task_recorder = task_recorder
        self.status_ok = True

    def run(self):
        while True:
            if self.task_queue.terminate():
                break
            task = self.task_queue.next_task()
            if task is None:
                time.sleep(5)
                continue
            logging.info("thread [%s] assigned cmd [%s]", self.tid, task.cmd)

            start_ts = get_current_timestring()
            success = shell_wrapper.shell_command(cmd=task.cmd, print_info=True)
            # success = True if os.system(task.cmd) == 0 else False

            if success:
                logging.info("successfully complete job [%s] [%s]", task.cid, task.cmd)
            else:
                if task.crucial:
                    logging.error("fail to complete crucial job [%s] [%s]", task.cid, task.cmd)
                else:
                    logging.warning("fail to complete negligible job [%s] [%s]", task.cid, task.cmd)

            end_ts = get_current_timestring()
            if self.task_recorder is not None:
                self.task_recorder.add(task.cmd, success, start_ts, end_ts)
            self.task_queue.finish_task(task.cid, success)

        logging.info("thread [%s] finished...", str(self.tid))


class TaskManager(object):
    def __init__(self):
        self.cmds = dict()
        self.thread_num_max = DEFAULT_MAX_THREAD
        self.queue = TopologyQueue()
        self.recoder = TaskRecorder()

    def bound_threads(self, max_num):
        if max_num > 0:
            self.thread_num_max = max_num

    def add_task(self, cid, cmd, is_crucial=True):
        self.queue.add_task(cid, cmd, is_crucial)

    def set_dependency(self, dependency_cid, following_cid):
        self.queue.set_dependency(dependency_cid, following_cid)

    def run(self):
        self.queue.sort()
        if self.queue.size() == 0:
            return
        task_runners = []
        for i in xrange(self.thread_num_max):
            task_runner = TaskRunner(i, self.queue, self.recoder)
            task_runner.setDaemon(True)
            task_runner.start()
            task_runners.append(task_runner)

        for task_runner in task_runners:
            task_runner.join()

    def status(self):
        return self.queue.status()

    def failed_stage(self):
        return self.queue.failed_stage()

    def record(self):
        return self.recoder.pool


if __name__ == "__main__":
    t = TaskManager()
    t.bound_threads(2)
    t.add_task("1", "echo 1", True)
    t.add_task("2", "echo 2", True)
    t.add_task("3", "echo 3", True)
    t.add_task("4", "echo 4", True)
    t.set_dependency("1", "2")
    t.set_dependency("1", "3")
    t.set_dependency("2", "4")
    t.set_dependency("3", "4")
    t.run()
    print >> sys.stdout, "job status:", t.status()
    for item in t.record():
        print >> sys.stdout, item.cmd, item.status, item.start_time, item.finish_time
