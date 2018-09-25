# -*- coding: utf-8 -*-

import re

import tensorflow as tf
import tensorflow.contrib.keras as kr
from sklearn.model_selection import train_test_split

from news_w2v.news_vec import NewsW2V
from seq2seq_attention_config import Seq2SeqAttentionConfig
from seq2seq_attention_model import Seq2SeqAttentionModel
from tagjieba.instance import TagJieba


class Seq2SeqAttentionTrain(object):
    def __init__(self):
        self.word2vec_model = NewsW2V().w2v_model
        self.word2vec_vocal_dict = dict(zip(self.word2vec_model.wv.index2word,
                                            range(len(self.word2vec_model.wv.index2word))))
        print('init Word2Vec model success....')

        self.config = Seq2SeqAttentionConfig()
        self.model = Seq2SeqAttentionModel(self.config, self.word2vec_model.wv.vectors)

        self.tag_jieba = TagJieba()
        print('init tag jieba success....')

    def word_to_id(self, words):
        """
        this function is able to get word id from word2vec vocals
        :param words: word list
        :return: word id list
        """
        words_id = []
        for word in words:
            try:
                words_id.append(self.word2vec_vocal_dict[word])
            except:
                pass

        return words_id

    def regular_content(self, content):
        """
        regular content, delete [content], website address, #content, (content)
        :param content: regular content
        :return: content by regular
        """
        # filter website address
        website_addresses = '.'.join(re.findall(u'\w*://.*', content))
        for website_address in website_addresses:
            content = content.replace(website_address, '')

        # filter chinese in bracket
        brackets = '.'.join(re.findall(r"（[\u4e00-\u9fff]+）", content))
        for bracket in brackets:
            content = content.replace(bracket, '')

        # filter #chinese
        channels = '.'.join(re.findall(u'#[\u4e00-\u9fff]+|#[\u4e00-\u9fff]+#', content))
        channels = channels.split('.')
        for channel in channels:
            content = content.replace(channel, '')

        # filter chinese and [chinese]
        expressions = '.'.join(re.findall(r'\[\w*[\u4e00-\u9fff]*\w*[\u4e00-\u9fff]*]', content))
        expressions = expressions.split('.')
        for expression in expressions:
            content = content.replace(expression, '')

        return content

    def read_data_set(self):
        """
        read data set
        :return: text and label
        """
        texts = open(self.config.text_path)
        labels = open(self.config.label_path)

        # regular
        texts = [self.regular_content(text) for text in texts]
        labels = [self.regular_content(label) for label in labels]

        # delete text which length is less than 200
        texts_ = []
        labels_ = []
        for index in range(len(texts)):
            if len(texts[index]) < 200:
                continue

            texts_.append(texts[index])
            labels_.append(labels[index])

        return texts_, labels_

    def batch_iter(self, X, y, batch_size=4):
        """
        this function is able to get batch iterate of total data set
        :param X: X
        :param y: y
        :param batch_size: batch size
        :return: batch iterate
        """
        data_len = len(X)

        num_batch = int((data_len - 1) / batch_size) + 1
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)

            yield X[start_id:end_id], y[start_id:end_id]

    def train(self):
        texts, labels = self.read_data_set()

        texts_words = [self.tag_jieba.cut(text) for text in texts]
        labels_words = [self.tag_jieba.cut(label) for label in labels]

        texts_words_id = [self.word_to_id(words) for words in texts_words]
        labels_words_id = [self.word_to_id(words) for words in labels_words]

        texts_words_id = kr.preprocessing.sequence.pad_sequences(texts_words_id, self.config.article_length)
        labels_words_id = kr.preprocessing.sequence.pad_sequences(labels_words_id, self.config.abstract_length)

        X_train, X_val, y_train, y_val = train_test_split(texts_words_id, labels_words_id, self.config.train_ds_rate)

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