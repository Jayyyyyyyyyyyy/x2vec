# -*- coding: utf-8 -*-

import sys
import os
import logging
import shell_wrapper
import glob

reload(sys)
sys.setdefaultencoding("utf-8")
sys.path.append(os.getcwd())


HADOOP_HOME = None
if os.environ.get("HADOOP_HOME", "") != "":
    HADOOP_HOME = os.environ["HADOOP_HOME"].strip()
assert HADOOP_HOME, "error! HADOOP_HOME not set!"


STREAMING_JAR = None
jars = glob.glob("{}/share/hadoop/tools/lib/hadoop-streaming*.jar".format(HADOOP_HOME))
if len(jars) > 0:
    STREAMING_JAR = sorted(jars)[-1]
assert STREAMING_JAR, "error! STREAMING_JAR not set"


class HadoopStreamingWrapper(object):
    def __init__(self):
        self._generic_options = []
        self._streaming_options = []

    def add_generic_option(self, _property, _value):
        _property = str(_property)
        _value = str(_value)
        if len(_property) == 0 or len(_value) == 0:
            raise Exception(
                "generic option -D <property> and <value> must not be empty")
        self._generic_options.append([_property, _value])

    def add_streaming_option(self, _property, _value):
        _property = str(_property)
        _value = str(_value)
        if len(_property) == 0 or len(_value) == 0:
            raise Exception(
                "streaming option <property> and <value> must not be empty")
        self._streaming_options.append([_property, _value])

    def clear(self):
        self._generic_options = []
        self._streaming_options = []

    def check_essentials(self):
        essentials = {"input": 1, "output": 1, "mapper": 1}
        for _property, _value in self._streaming_options:
            if _property in essentials:
                del essentials[_property]
        if len(essentials) > 0:
            raise Exception(
                "missing streaming_options: %s" % essentials.keys())

    def build_cmd(self):
        self.check_essentials()
        cmd = "%s/bin/hadoop jar %s" % (HADOOP_HOME, STREAMING_JAR)
        for _property, _value in self._generic_options:
            tmp_value = _value
            if " " in _value:
                tmp_value = "\"%s\"" % _value
            cmd += " -D %s=%s" % (_property, tmp_value)
        for _property, _value in self._streaming_options:
            tmp_value = _value
            if " " in _value:
                tmp_value = "\"%s\"" % _value
            cmd += " -%s %s" % (_property, tmp_value)
        return cmd

    def run(self, **kwargs):
        cmd = self.build_cmd()
        if kwargs.get("print_cmd", False):
            logging.info(cmd)
        return shell_wrapper.shell_command(
            cmd=cmd,
            print_info=kwargs.get("print_info", False))


HadoopStreaming = HadoopStreamingWrapper()

# python hadoop_streaming_wrapper.py
if __name__ == "__main__":
    logging.basicConfig(
            level = logging.INFO,
            format = '[%(asctime)s - %(filename)s - %(levelname)s] %(message)s',
            datefmt = '%Y-%m-%d %H:%M:%S')
    hs = HadoopStreamingWrapper()
    hs.add_streaming_option("input", "/user/hadoop/file1")
    hs.add_streaming_option("input", "/user/hadoop/file2")
    hs.add_streaming_option("output", "/user/hadoop/dir1")
    hs.add_streaming_option("mapper", "cat")
    print hs.run(print_cmd=True)
