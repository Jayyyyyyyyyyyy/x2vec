# -*- coding: utf-8 -*-

import sys
import os
import logging
import subprocess

reload(sys)
sys.path.append(os.getcwd())


def shell_command(**kwargs):
    try:
        proc = subprocess.Popen(kwargs['cmd'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        if kwargs.get("print_info", False):
            with proc.stdout:
                for line in iter(proc.stdout.readline, b''):
                    logging.info(line.strip())
        proc.wait()
        return proc.returncode == 0
    except KeyboardInterrupt:
        logging.exception("killed by KeyboardInterrupt")
    return False


def shell_command_stdout(**kwargs):
    try:
        proc = subprocess.Popen(kwargs['cmd'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        with proc.stdout:
            for line in iter(proc.stdout.readline, b''):
                yield line.strip()
        proc.wait()
    except KeyboardInterrupt:
        logging.exception("killed by KeyboardInterrupt")


if __name__ == "__main__":
    pass
