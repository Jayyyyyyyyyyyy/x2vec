#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random


class ReservoirSampling(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.passed = 0
        self.data = []

    def add(self, ele):
        if self.passed < self.capacity:
            self.data.append(ele)
        else:
            replace = random.randint(0, self.passed - 1)
            if replace < self.capacity:
                self.data[replace] = ele
        self.passed += 1

    def size(self):
        return len(self.data)
