# -*- coding: utf-8 -*-

import tensorflow as tf

from seq2seq_attention_config import Seq2SeqAttentionDataConfig
from seq2seq_attention_config import Seq2SeqAttentionModelConfig
from seq2seq_attention_model import Seq2SeqAttentionModel
from batch_reader import BatchReader
from data_loader import DataLoader

class Seq2SeqAttentionTrain(object):
    def __init__(self):
        self.model_config = Seq2SeqAttentionModelConfig()
        self.data_config = Seq2SeqAttentionDataConfig()

        self.data_loader = DataLoader(self.data_config)
        self.batch_reader = BatchReader(self.model_config, self.data_config, self.data_loader)
        print('over')

        self.model = Seq2SeqAttentionModel(self.model_config, self.data_loader.word2vec_vectors)

    def train(self):
        """
        train model
        :return:
        """
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print('coming....')
            continue_train = True
            while continue_train:
                batch = self.batch_reader.next_batch()
                if batch is None:
                    break

                encoder_outputs = session.run(self.model.encoder_outputs, feed_dict={self.model.article: batch[0],
                                                                                     self.model.abstract: batch[1]})
                print(encoder_outputs)

if __name__ == '__main__':
    seq2seq_attention_train = Seq2SeqAttentionTrain()
    seq2seq_attention_train.train()