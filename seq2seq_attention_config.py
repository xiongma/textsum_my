# -*- coding: utf-8 -*-
import os

file_path = os.path.dirname(os.path.realpath(__file__))

class Seq2SeqAttentionModelConfig(object):
    def __init__(self):
        self.batch_size = 4
        self.article_length = 20
        self.cell_output_size = 256
        self.encoder_layer_num = 4
        self.cell_output_prob = 0.5
        self.abstract_length = 5
        self.num_softmax_samples = 4096
        self.train_ds_rate = 0.8
        self.should_stop = 100      # if 100 time not improve,stop

        # bucket
        self.queue_num_batch = 100
        self.bucket_cache_batch = 10
        self.bucketing = False

class Seq2SeqAttentionDataConfig(object):
    def __init__(self):
        self.sentence_start = '<s>'
        self.sentence_end = '</s>'
        self.pad_token = '<PAD>'

        self.text_path = file_path + r'/data/test_text.txt'
        self.label_path = file_path + r'/data/test_label.txt'
