# -*- coding: utf-8 -*-

class Seq2SeqAttentionConfig(object):
    def __init__(self):
        self.batch_size = 32
        self.sequence_length = 200
        self.cell_output_size = 256
        self.encoder_layer_num = 4
        self.cell_output_prob = 0.5