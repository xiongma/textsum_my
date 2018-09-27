# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.keras as kr
from sklearn.model_selection import train_test_split

from seq2seq_attention_config import Seq2SeqAttentionDataConfig
from seq2seq_attention_config import Seq2SeqAttentionModelConfig
from seq2seq_attention_model import Seq2SeqAttentionModel
from batch_reader import BatchReader
from data_loader import DataLoader


class Seq2SeqAttentionTrain(object):
    def __init__(self):
        self.data_loader = DataLoader(self.data_config)

        self.model_config = Seq2SeqAttentionModelConfig()
        self.data_config = Seq2SeqAttentionDataConfig()
        self.model = Seq2SeqAttentionModel(self.model_config, self.data_loader.word2vec_model.wv.vectors)
        self.batch_reader = BatchReader(self.model_config, self.data_config)

    def train(self):
        """
        train model
        :return:
        """
        texts, labels = self.read_data_set()

        texts_words = [self.tag_jieba.cut(text) for text in texts]
        labels_words = [self.tag_jieba.cut(label) for label in labels]

        texts_words_id = [self.word_to_id(words) for words in texts_words]
        labels_words_id = [self.word_to_id(words) for words in labels_words]

        texts_words_id = kr.preprocessing.sequence.pad_sequences(texts_words_id, self.model_config.article_length)
        labels_words_id = kr.preprocessing.sequence.pad_sequences(labels_words_id, self.model_config.abstract_length)

        X_train, X_val, y_train, y_val = train_test_split(texts_words_id, labels_words_id, self.model_config.train_ds_rate)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            batch_iterate = self.batch_iter(X_train, y_train)
            for texts, labels in batch_iterate:
                encoder_outputs = session.run(self.model.encoder_outputs, feed_dict={self.model.article: texts,
                                                               self.model.abstract: labels})
                print(encoder_outputs)

if __name__ == '__main__':
    seq2seq_attention_train = Seq2SeqAttentionTrain()
    seq2seq_attention_train.train()