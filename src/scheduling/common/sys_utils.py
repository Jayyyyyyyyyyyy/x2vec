# -*- coding: utf-8 -*-

import sys
import os
import socket

reload(sys)
sys.path.append(os.getcwd())


def get_host_name():
    return socket.gethostname()
