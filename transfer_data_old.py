
# import tensorflow as tf
# import six
#
text_path = "/Users/maxiong/Workpace/Code/Python/textsum_my/data/train_text.txt"
label_path = "/Users/maxiong/Workpace/Code/Python/textsum_my/data/train_label.txt"
out_path = "/Users/maxiong/Workpace/Code/Python/textsum_my/data/data_set1.txt"
out_vocab_path = "/Users/maxiong/Workpace/Code/Python/textsum_my/data/vocab.txt"
# label_path = "/Users/maxiong/Workpace/Code/Python/textsum_my/data/test_label.txt"
# out_path = "/Users/maxiong/Workpace/Code/Python/textsum_my/data/test_label.tfrecords"
# text_file = open(text_path, mode='r')
# label_file = open(label_path, mode='r')
# writer = tf.python_io.TFRecordWriter(out_path)
# for text, label in zip(text_file, label_file):
#     example = tf.train.Example(features=tf.train.Features(feature={
#         "article": tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode(encoding='utf-8')])),
#         "abstract": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode(encoding='utf-8')])),
#         "publisher": tf.train.Feature(bytes_list=tf.train.BytesList(value=["AFP".encode(encoding='utf-8')]))
#     }))
#     writer.write(example.SerializeToString())
#
# writer.close()
# example_gen = ExampleGen(out_path)
# text = six.next(example_gen)
# print(str(text))
import re

def regular_content(content):
    """
    regular content, delete [content], website address, #content, (content)
    :param content: regular content
    :return: content by regular
    """
    content = content.replace(' ', '')
    # filter website address
    website_addresses = '.'.join(re.findall(u'\w*://.*', content))
    for website_address in website_addresses:
        content = content.replace(website_address, '')

    # filter chinese in bracket
    brackets = '.'.join(re.findall(r"（[\u4e00-\u9fff]+）", content))
    for bracket in brackets:
        content = content.replace(bracket, '')

    # filter #chinese
    channels = '.'.join(re.findall(u'[#*@*][\u4e00-\u9fff]*|[#*@*]', content))
    channels = channels.split('.')
    for channel in channels:
        content = content.replace(channel, '')

    # filter chinese and [chinese]
    expressions = '.'.join(re.findall(r'\[\w*[\u4e00-\u9fff]*\w*[\u4e00-\u9fff]*]', content))
    expressions = expressions.split('.')
    for expression in expressions:
        content = content.replace(expression, '')

    return content

def transfer_text():
    """
    transfer text to standard text
    :return:
    """
    text_file = open(text_path, "r")
    label_file = open(label_path, "r")

    out_file = open(out_path, 'w')

    for text, label in zip(text_file, label_file):
        text.strip('\t')
        text = text.replace('\n', '')
        text = text.replace('=', '')
        label = label.replace('\n', '')

        text = regular_content(text)
        label = regular_content(label)

        text = 'article=<d><p><s>' + text
        text = text.replace('。', '。</s><s>')
        text = text + '</s></p></d>'

        label = 'abstract=<d><p><s>' + label + '</s></p></d>'

        # text_label = standard.replace('value: ""', 'value: "'+text+'"', 1)
        # text_label = text_label.replace('value: ""', 'value: "'+label + '"', 1)

        final_text = 'publisher=AFP\t'+text+'\t'+label+'\n'

        out_file.write(final_text)

    out_file.close()

def bulid_vocab():
    in_file = open('/Users/maxiong/Workpace/Code/Python/textsum_my/data/tag_dict.txt', 'r')
    out_file = open(out_vocab_path, 'w')
    for line in in_file:
        word_parma = line.split(' ')
        out_file.write(word_parma[0]+' '+word_parma[1]+'\n')

    out_file.write('<PAD> 5')
    out_file.write('<UNK> 5')
    out_file.write('<d> 5')
    out_file.write('<s> 5')
    out_file.write('<p> 5')
    out_file.write('</d> 5')
    out_file.write('</s> 5')
    out_file.write('</p> 5')

    out_file.close()

bulid_vocab()
