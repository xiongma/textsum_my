# # -*- coding: utf-8 -*-
#
import tensorflow as tf
from tensorflow.contrib import rnn
#
def create_gru_unit(gru_hidden_size, gru_output_keep_prob, name_scope=None):
    """
    create gru unit
    :param gru_hidden_size: GRU output hidden_size
    :param gru_output_keep_prob: GRU output keep probability
    :param name_scope: GRU name scope
    :return: GRU cell
    """
    with tf.name_scope(name_scope):
        gru_cell = rnn.LSTMCell(gru_hidden_size)
        gru_cell = rnn.DropoutWrapper(cell=gru_cell, input_keep_prob=1.0,
                                      output_keep_prob=gru_output_keep_prob)

    return gru_cell
#
a = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
# a = [[1.0, 2.0], [3.0, 4.0]]
#
# b = tf.concat(axis=1, values=a)
# session = tf.Session()
# session.run(tf.global_variables_initializer())
# print(session.run(b))

# a_ = tf.unstack(tf.transpose(a))
sequence_length = tf.placeholder(tf.int32, [2])
input = tf.placeholder(tf.int32, [2, 2, 2])
# sequence_length = tf.constant([2])
gru_1 = create_gru_unit(5, 0.5, 'gru_1')
gru_2 = create_gru_unit(5, 0.5, 'gru_2')
outputs, fw_state, bw_state = tf.contrib.rnn.static_bidirectional_rnn(gru_1, gru_2, input, sequence_length=input.get_shape()[0], dtype=tf.float32)
session = tf.Session()
session.run(tf.global_variables_initializer())
print(session.run(outputs, feed_dict={input: a, sequence_length: 2}))

# import re
#
# def regular_content(content):
#     """
#     regular content, delete [content], website address, #content#
#     :param content: regular content
#     :return: content by regular
#     """
#     # text
#
#     website_address = '.'.join(re.findall(u'\w*://.*', content))
#     content = content.replace(website_address, '')
#
#     brackets = '.'.join(re.findall(r"（[\u4e00-\u9fff]+）", content))
#     for bracket in brackets:
#         content = content.replace(bracket, '')
#
#     channels = '.'.join(re.findall(u'#[\u4e00-\u9fff]+', content))
#     channels = channels.split('.')
#     for channel in channels:
#         content = content.replace(channel, '')
#
#     expressions = '.'.join(re.findall(r'[\u4e00-\u9fff]*\[[\u4e00-\u9fff]*\w*]', content))
#     expressions = expressions.split('.')
#     for expression in expressions:
#         content = content.replace(expression, '')
#
#     return content
#
# text = regular_content("""
#         在青海初麻乡中心学校，有一个叫冶卓的女孩，她的家离校有6小时的山路，而重病妈妈和残疾爸爸都无法相伴.只学了一年汉语的冶卓，各科成绩都名列前茅。家务和学习占据了她所有的时间，只有一个人时，歌谣和口琴声才会轻轻响起，小女孩“音乐家”的梦想才会微微发光...戳腾讯公益点亮冶卓的梦想[笑cry] #助学圆梦#江苏新闻的秒拍视频 http://t.cn/R9vSpM8""")
#
# print(text)
