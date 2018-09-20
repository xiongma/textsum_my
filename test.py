# -*- coding: utf-8 -*-

import tensorflow as tf

a = [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]]

b = tf.concat(axis=1, values=a)
session = tf.Session()
session.run(tf.global_variables_initializer())
print(session.run(b))