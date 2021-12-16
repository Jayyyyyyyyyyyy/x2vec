#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
sys.path.append(".")
import tensorflow as tf
import numpy as np
from sparse_values import SparseValues
from reservoir_sampling import ReservoirSampling
import hadoop_shell_wrapper as hadoop
import random


FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def dump_record(writer,
                diu_list,
                positives,
                play_seq_st):

    play_seq_st_tf = play_seq_st.to_tensor_value()
    diu_bytes = bytes("\x01".join(diu_list))

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "diu_list":
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[diu_bytes])
                    ),
                "positives": tf.train.Feature(int64_list=tf.train.Int64List(value=positives)),
                "play_ind":
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[play_seq_st_tf.indices.astype(np.int32).tobytes()])
                    ),
                "play_val":
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[play_seq_st_tf.values.astype(np.int32).tobytes()])
                    )
            }
        )
    )
    writer.write(example.SerializeToString())


def parse_feature(cols,
                  vocab_vid_size,
                  diu_list,
                  positives,
                  batch_record_idx,
                  play_seq_st):

    diu_list.append(cols[0])
    positives.append(int(cols[1]))
    for x in cols[2].split(","):
        f = int(x)
        assert 0 <= f < vocab_vid_size
        play_seq_st.add(batch_record_idx, f, f)


def reducer(args):

    input_partname = str(os.environ["mapreduce_task_partition"])
    assert input_partname
    input_partname = "part-{}".format(input_partname.zfill(5))

    batch_size = args.batch_size
    vocab_vid_size = args.vocab_vid_size
    output_path = args.output_path
    assert batch_size > 0
    assert vocab_vid_size > 0
    assert output_path

    backup = ReservoirSampling(batch_size)

    play_seq_st = SparseValues((batch_size, vocab_vid_size))
    positives = []
    diu_list = []

    batch_record_idx = 0
    record_num = 0

    local_tfrecord_filename = "{}/{}.tfrecord".format(FILE_DIR, input_partname)

    writer = tf.io.TFRecordWriter(local_tfrecord_filename)

    for line in sys.stdin:
        cols = line.strip().split("\t")
        # hash | diu | positive | play_seq
        if len(cols) != 4:
            print >> sys.stderr, "error cols format:", cols
            continue

        backup.add(cols)

        # 已经累积batch_size，可以输出一个tfrecord
        if batch_record_idx == batch_size:
            dump_record(writer,
                        diu_list if args.keep_diu else ["null"],
                        positives,
                        play_seq_st)
            record_num += 1

            # 重置batch data
            batch_record_idx = 0
            positives = []
            diu_list = []
            play_seq_st.clear()

        # diu = cols[1]
        parse_feature(cols[1:],
                      vocab_vid_size,
                      diu_list,
                      positives,
                      batch_record_idx,
                      play_seq_st)

        batch_record_idx += 1
        # end of foreach sys.stdin

    # 要求数据量不可以小于1个batch
    assert record_num > 0

    # 如果最后一个batch不满batch_size，就随机补齐
    last_batch_size = batch_record_idx
    if batch_record_idx < batch_size:
        short_num = batch_size - batch_record_idx
        for cols in backup.data[:short_num]:
            parse_feature(cols[1:],
                          vocab_vid_size,
                          diu_list,
                          positives,
                          batch_record_idx,
                          play_seq_st)
            batch_record_idx += 1
    assert batch_record_idx == batch_size

    dump_record(writer,
                diu_list if args.keep_diu else ["null"],
                positives,
                play_seq_st)
    record_num += 1

    writer.close()

    if not hadoop.put(local_tfrecord_filename, args.output_path):
        print >> sys.stderr, "fail to put {} to {}".format(local_tfrecord_filename, output_path)
        return 1
    print >> sys.stdout, "batch_num={}, batch_size={}, last_batch_size={}".format(
        record_num, batch_size, last_batch_size)
    return 0


def mapper(args):
    for line in sys.stdin:
        print >> sys.stdout, "{}\t{}".format(random.randint(0, 10000000), line.strip())
    return 0


def run(args):
    return mapper(args) if args.stage == "map" else reducer(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build samples')
    parser.add_argument("--stage", dest='stage', type=str, required=True, choices=["map", "reduce"], help="map/reduce")
    parser.add_argument("--batch_size", dest='batch_size', type=int, default=0, required=False, help="batch size")
    parser.add_argument("--vocab_vid_size", dest='vocab_vid_size', type=int, default=0, required=False, help="vocab_vid_size")
    parser.add_argument("--output_path", dest='output_path', type=str, default="", required=False, help="output_path")
    parser.add_argument("--keep_diu", dest='keep_diu', action="store_true", help="whether keep diu feature")
    arguments = parser.parse_args()

    try:
        sys.exit(run(arguments))
    except Exception as ex:
        print >> sys.stderr, ex
        sys.exit(1)
