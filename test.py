# !/usr/bin python3
# coding=utf-8
import re
import os
from tagjieba.instance import TagJieba
import codecs
import numpy as np

def clean_and_segmentate_data(raw_data_list, cwd, decoding='GB18030'):
    tag_jieba = TagJieba()
    for raw_data in raw_data_list:
        out_data = "clean." + raw_data
        raw_data = cwd + raw_data
        print("|||", out_data)

# 注意！搜狗实验室下载的数据编码是 [GB18030]，注意！
        # sogou_raw_data [news_tensite_xml*] use [gb18030]
        with codecs.open(raw_data, 'r', decoding) as raw:
            with open(out_data, 'w') as out:
                abstract = ""
                for i, line in enumerate(raw):
                    i %= 6
                    # filter punctuation(full-width characters, half-width characters)
                    # except full-width comma[，]
                    if 3 <= i <= 4:
                        line = re.sub(r'（.*）', '', line)
                        line = re.sub(r'\(.*\)', '', line)
                        line = re.sub("[\s+.\!\/_,$%^*(+\"\']+|[+——！。？、~@#￥%……&*（）＂＂“”]+", "", line)
# 上面过滤掉 (图 曾伊言 摄) 之类的语句
# 上面过滤掉 全角半角标点符号，而 textsum 是用 <s></s> 作为句段的分隔标记的

                    if i == 3:
                        line = ' '.join(tag_jieba.cut(line[14:-14]))
                        abstract = "abstract=<d> <p> <s> %s</s> </p> </d>" % line
                    elif i == 4:

                        line = ' '.join(tag_jieba.cut(line[9:-9]))
                        article = "article=<d> <p> <s> %s </s> </p> </d>" % line.replace('，', '</s> <s> ')


# 如果输入的文章过长，那么处理效率会比较低，甚至运行会报错
# 考虑到人类的文章开头包含更多和摘要相关的内容，所以可以对文章进行截取 article[:256]
# if len(article) > 64 过滤掉正文过短的文章
                        if len(article) > 64:  # prevent the article shorter than abstract
                            temp = "publisher=AFP\t%s\t%s\n" % (abstract, article[:256])
                            print(temp, end='')
                            out.write(temp)

clean_and_segmentate_data(['news_tensite_xml.dat'], '/Users/maxiong/Workpace/Code/Python/textsum_my/data/')