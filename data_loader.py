
import re

import numpy as np
from six.moves import xrange

from news_w2v.news_vec import NewsW2V
from tagjieba.instance import TagJieba

class DataLoader(object):
    def __init__(self, data_config):
        self.data_config = data_config
        self.word2vec_model = NewsW2V().w2v_model
        self.word2vec_vocal_dict = dict(zip(self.word2vec_model.wv.index2word,
                                            range(len(self.word2vec_model.wv.index2word))))

        # add <PAD>, <s>, </s>, <None>
        self.word2vec_vocal_dict[self.data_config.pad_token] = len(self.word2vec_vocal_dict)
        self.word2vec_vocal_dict[self.data_config.sentence_start] = len(self.word2vec_vocal_dict)
        self.word2vec_vocal_dict[self.data_config.sentence_end] = len(self.word2vec_vocal_dict)
        self.word2vec_vectors = self.word2vec_model.wv.vectors.tolist()
        for i in range(3):
            self.word2vec_vectors.append(np.random.uniform(-1, 1, size=len(self.word2vec_model.wv.vectors[0])))
        print('init Word2Vec model success....')

        self.tag_jieba = TagJieba()
        print('init tag jieba success....')

    def read_data_set(self):
        """
        read data set
        :return: text and label
        """
        texts = open(self.data_config.text_path)
        labels = open(self.data_config.label_path)

        # regular
        texts = [self.regular_content(text) for text in texts]
        labels = [self.regular_content(label) for label in labels]

        # delete text which length is less than 200
        texts_ = []
        labels_ = []
        for index in range(len(texts)):
            if len(texts[index]) < 100:
                continue

            texts_.append(texts[index])
            labels_.append(labels[index])

        return texts_, labels_

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

    def word_to_id(self, word):
        """
        this function is able to get word id from word2vec vocals
        :param word: word list
        :return: word
        """
        try:
            return self.word2vec_vocal_dict[word]

        except:
            # """
            #     if word not in word2vec vocal dict, return None id
            # """
            # return len(self.word2vec_vocal_dict)
            pass

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
        for i in xrange(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)

            yield X[start_id:end_id], y[start_id:end_id]