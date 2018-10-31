# -*- coding: utf-8 -*-
import os

file_path = os.path.dirname(os.path.realpath(__file__))

class Seq2SeqAttentionModelConfig(object):
    def __init__(self):
        self.batch_size = 4
        self.article_length = 20    # encode time steps
        self.cell_output_size = 256
        self.encoder_layer_num = 4
        self.cell_output_prob = 0.5
        self.abstract_length = 5    # decode time steps
        self.num_softmax_samples = 4096
        self.train_ds_rate = 0.8
        self.max_step = 1000000      # if 100 time not improve,stop
        self.beam_size = 8
        self.min_lr_rate = 0.01
        self.max_grad_norm = 2

        # bucket
        self.queue_num_batch = 100
        self.bucket_cache_batch = 10
        self.bucketing = False

        # op
        self.model = 'train'

        # model and log path
        self.model_path = '/model/'
        self.log_path = '/log/'
        self.train_dir = '/train/'
        self.eval_dir = '/eval/'
        # how often to save the model
        self.save_model_secs = 600

        # number of gpu
        self.num_gpu = 0

class Seq2SeqAttentionDataConfig(object):
    def __init__(self):
        self.sentence_start = '<s>'
        self.sentence_end = '</s>'
        self.pad_token = '<PAD>'

        self.text_path = file_path + r'/data/test_text.txt'
        self.label_path = file_path + r'/data/test_label.txt'
