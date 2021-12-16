#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class SparseValues(object):
    def __init__(self, dense_shape):
        self.__dense_shape = np.asarray(dense_shape, dtype=np.int64)
        self.__values = []

    def add(self, row, col, value):
        self.__values.append((row, col, value))

    def clear(self):
        self.__values = []

    def to_tensor_value(self):
        # most ops assume correct ordering
        self.__values.sort()

        indices = []
        values = []
        for row, col, value in self.__values:
            indices.append((row, col))
            values.append(value)

        # other sparse API cannot handle "all 0" tensor
        # so to work around, for 'all 0' tensor, put in a zero element
        # sacrifice the performance a little bit
        if len(indices) == 0:
            indices.append((0, 0))
            values.append(0)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=np.int64)
        return tf.compat.v1.SparseTensorValue(
            indices=indices,
            values=values,
            dense_shape=self.__dense_shape)
