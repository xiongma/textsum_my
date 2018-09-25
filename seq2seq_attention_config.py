# -*- coding: utf-8 -*-
import os

file_path = os.path.dirname(os.path.realpath(__file__))

class Seq2SeqAttentionConfig(object):
    def __init__(self):
        self.batch_size = 32
        self.article_length = 200
        self.cell_output_size = 256
        self.encoder_layer_num = 4
        self.cell_output_prob = 0.5
        self.abstract_length = 30
        self.num_softmax_samples = 4096
        self.train_ds_rate = 0.8

        self.text_path = file_path + r'/data/text.txt'
        self.label_path = file_path + r'/data/label.txt'