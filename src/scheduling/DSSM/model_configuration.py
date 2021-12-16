#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging


class ModelConfiguration(object):

    def __init__(self):
        self.batch_size = 0
        self.vocab_vid_size = 0
        self.embed_vid_size = 0
        self.seq_max_size = 0
        self.epoches = 0
        self.gpu_core = 0
        self.negative_samples = 0
        self.l2_reg_rate = 0.0
        self.model_keep = 1
        self.checkpoint_steps = 0
        self.learning_rate = 0
        self.fc_layers = []
        self.triplet_margin = -1
        self.eval_step = 0
        self.test_fraction = 0

    def check(self):
        return self.batch_size > 0 and \
               self.vocab_vid_size > 0 and \
               self.embed_vid_size > 0 and \
               self.seq_max_size > 0 and \
               self.epoches > 0 and \
               self.gpu_core >= 0 and \
               self.negative_samples > 0 and \
               self.model_keep > 0 and \
               self.checkpoint_steps > 0 and \
               self.learning_rate > 0 and \
               len(self.fc_layers) > 0 and \
               self.triplet_margin > 0

    def load(self, filename):
        with open(filename, "r") as fin:
            file_content = fin.read()
            parsed_conf = json.loads(file_content)
            self.batch_size = parsed_conf["batch_size"]
            self.vocab_vid_size = parsed_conf["vocab_vid_size"]
            self.embed_vid_size = parsed_conf["embed_vid_size"]
            self.seq_max_size = parsed_conf["seq_max_size"]
            self.epoches = parsed_conf["epoches"]
            self.gpu_core = parsed_conf["gpu_core"]
            self.negative_samples = parsed_conf["negative_samples"]
            self.l2_reg_rate = parsed_conf["l2_reg_rate"]
            self.model_keep = parsed_conf["model_keep"]
            self.checkpoint_steps = parsed_conf["checkpoint_steps"]
            self.learning_rate = parsed_conf["learning_rate"]
            self.fc_layers = parsed_conf["fc_layers"]
            self.triplet_margin = parsed_conf["triplet_margin"]
            self.eval_step = parsed_conf["eval_step"]
            self.test_fraction = parsed_conf["test_fraction"]
        return self

    def persist(self, filename):
        with open(filename, "w") as fout:
            config_map = {
                "batch_size": self.batch_size,
                "embed_vid_size": self.embed_vid_size,
                "vocab_vid_size": self.vocab_vid_size,
                "seq_max_size": self.seq_max_size,
                "epoches": self.epoches,
                "gpu_core": self.gpu_core,
                "negative_samples": self.negative_samples,
                "l2_reg_rate": self.l2_reg_rate,
                "model_keep": self.model_keep,
                "checkpoint_steps": self.checkpoint_steps,
                "learning_rate": self.learning_rate,
                "fc_layers": self.fc_layers,
                "triplet_margin": self.triplet_margin,
                "eval_step": self.eval_step,
                "test_fraction": self.test_fraction
            }
            conf_content = json.dumps(config_map, ensure_ascii=False, indent=4)
            fout.write(conf_content)

    def show(self):
        logging.info("===== model configuration =====")
        logging.info("batch_size = %d", self.batch_size)
        logging.info("embed_vid_size = %d", self.embed_vid_size)
        logging.info("vocab_vid_size = %d", self.vocab_vid_size)
        logging.info("seq_max_size = %d", self.seq_max_size)
        logging.info("epoches = %d", self.epoches)
        logging.info("gpu_core = %d", self.gpu_core)
        logging.info("negative_samples = %d", self.negative_samples)
        logging.info("l2_reg_rate = %f", self.l2_reg_rate)
        logging.info("model_keep = %d", self.model_keep)
        logging.info("checkpoint_steps = %d", self.checkpoint_steps)
        logging.info("learning_rate = %f", self.learning_rate)
        logging.info("fc_layers = %s", repr(self.fc_layers))
        logging.info("triplet_margin = %f", self.triplet_margin)
        logging.info("eval_step = %d", self.eval_step)
        logging.info("test_fraction = %f", self.test_fraction)
