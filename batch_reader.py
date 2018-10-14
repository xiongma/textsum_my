
import time
from collections import namedtuple
from random import shuffle
from threading import Thread

from seq2seq_attention_config import Seq2SeqAttentionDataConfig, Seq2SeqAttentionModelConfig
from data_loader import DataLoader

import numpy as np
import six
from six.moves import queue as Queue
from six.moves import xrange

"""
这里input_queue的作用是将数据集一个一个的放入，放入的是文本和摘要
    bucket_input_queue的作用是将数据集切成一个一个的batch_size大小，然后在用的时候取出来,取出来的时候是一个一个batch取出来
"""
class BatchReader(object):
    def __init__(self, model_config, data_config, data_loader):
        self.model_config = model_config
        self.data_config = data_config
        self.data_loader = data_loader

        self.model_input = namedtuple('ModelInput',
                        'enc_input dec_input target enc_len dec_len '
                        'origin_article origin_abstract')

        self.input_queue = Queue.Queue(self.model_config.bucket_cache_batch * self.model_config.batch_size)
        self.bucket_input_queue = Queue.Queue(self.model_config.queue_num_batch)

        # input threads: data set input thread
        self.input_threads = []
        for _ in xrange(1):
            self.input_threads.append(Thread(target=self.fill_input_queue()))
            self.input_threads[-1].daemon = True
            self.input_threads[-1].start()

        """
            bucketing threads: data set is used to split data set by batch size, then when use data set, 
            get data set from this queue
        """
        self.bucketing_threads = []
        for x in range(1):
            self.bucketing_threads.append(Thread(target=self.fill_bucket_into_queue))
            self.bucketing_threads[-1].daemon = True
            self.bucketing_threads[-1].start()

        """
            daemon thread, use to watch input threads and bucketing threads, if they die, use new thread instead of them
        """
        # self.watch_thread = Thread(target=self.watch_threads)
        # self.watch_thread.daemon = True
        # self.watch_thread.start()

    def next_batch(self):
        """
        get next batch of data set
        :return:
                enc_batch: A batch of encoder inputs [batch_size, article_length].
                dec_batch: A batch of decoder inputs [batch_size, abstract_length].
                target_batch: A batch of targets [batch_size, abstract_length].
                enc_input_len: encoder input lengths of the batch.
                dec_input_len: decoder input lengths of the batch.
                loss_weights: weights for loss function, 1 if not padded, 0 if padded.
                origin_articles: original article words.
                origin_abstracts: original abstract words.
        """
        enc_batch = np.zeros((self.model_config.batch_size, self.model_config.article_length), dtype=np.int32)
        enc_input_lens = np.zeros((self.model_config.batch_size), dtype=np.int32)
        dec_batch = np.zeros((self.model_config.batch_size, self.model_config.abstract_length), dtype=np.int32)
        dec_output_lens = np.zeros((self.model_config.batch_size), dtype=np.int32)
        target_batch = np.zeros((self.model_config.batch_size, self.model_config.abstract_length), dtype=np.int32)
        loss_weights = np.zeros((self.model_config.batch_size, self.model_config.abstract_length), dtype=np.float32)

        origin_articles = ['None'] * self.model_config.batch_size
        origin_abstracts = ['None'] * self.model_config.batch_size

        buckets = self.bucket_input_queue.get()
        print(len(buckets))
        for i in range(self.model_config.batch_size):
            (enc_inputs, dec_inputs, targets, enc_input_len, dec_output_len,
             article, abstract) = buckets[i]

            origin_articles[i] = article
            origin_abstracts[i] = abstract
            enc_input_lens[i] = enc_input_len
            dec_output_lens[i] = dec_output_len
            enc_batch[i, :] = enc_inputs[:]
            dec_batch[i, :] = dec_inputs[:]
            target_batch[i, :] = targets[:]
            for j in xrange(dec_output_len):
                loss_weights[i][j] = 1

        return (enc_batch, dec_batch, target_batch, enc_input_lens, dec_output_lens,
                loss_weights, origin_articles, origin_abstracts)

    def fill_input_queue(self):
        """
        fill data set into input queue one by one
        :return:
        """
        articles, abstracts = self.data_loader.read_data_set()
        """
            fill input queue one by one
        """
        batch_iter = self.data_loader.batch_iter(articles, abstracts, self.model_config.batch_size)

        start_id = self.data_loader.word_to_id(self.data_config.sentence_start)
        end_id = self.data_loader.word_to_id(self.data_config.sentence_end)
        pad_id = self.data_loader.word_to_id(self.data_config.pad_token)

        while True:
            """
                batch_iter is a generator, each element is article and abstract
            """
            try:
                article_batch, abstract_batch = six.next(batch_iter)

                for article, abstract in zip(article_batch, abstract_batch):
                    article_words = self.data_loader.tag_jieba.cut(article, delete_stop_words=True)
                    abstract_words = self.data_loader.tag_jieba.cut(abstract, delete_stop_words=True)

                    enc_inputs = []
                    dec_inputs = [start_id]
                    """
                        if article length or abstract length greater than config length, pad to config length
                        if article length or abstract length less than config length, use itself length
                    """
                    for i in xrange(min(len(article_words), self.model_config.article_length)):
                        word_id = self.data_loader.word_to_id(article_words[i])
                        if word_id is None:
                            continue
                        enc_inputs.append(word_id)

                    for i in xrange(min(len(abstract_words), self.model_config.abstract_length-1)):
                        word_id = self.data_loader.word_to_id(abstract_words[i])
                        if word_id is None:
                            continue
                        dec_inputs.append(self.data_loader.word_to_id(abstract_words[i]))

                    targets = dec_inputs[1:]
                    targets.append(end_id)

                    # PAD if necessary
                    while len(enc_inputs) < self.model_config.article_length:
                        enc_inputs.append(pad_id)

                    while len(dec_inputs) < self.model_config.abstract_length:
                        dec_inputs.append(end_id)

                    while len(targets) < self.model_config.abstract_length:
                        targets.append(end_id)

                    element = self.model_input(enc_inputs, dec_inputs, targets, len(enc_inputs), len(targets),
                                               ' '.join(article), ' '.join(abstract))

                    self.input_queue.put(element)

            except StopIteration:
                print("data set fill input queue success....")
                break

    def fill_bucket_into_queue(self):
        """
        split data set by every batch size, and then fill bucketed batches into the bucket_input_queue.
        :return:
        """
        while True:
            inputs = []
            queue_size = self.input_queue.qsize()
            k = 0
            for _ in range(queue_size):
                model_input = self.input_queue.get()
                inputs.append(model_input)

            # whether use bucket
            if self.model_config.bucketing:
                inputs = sorted(inputs, key=lambda inp: inp.enc_len)

            # split total batch by batch size
            batches = []
            for i in range(0, len(inputs), self.model_config.batch_size):
                batches.append(inputs[i:i+self.model_config.batch_size])

            # put batch into bucket queue
            shuffle(batches)
            for b in batches:
                self.bucket_input_queue.put(b)

    def watch_threads(self):
        """
        watch the daemon input threads and restart if dead.
        :return:
        """
        while True:
            time.sleep(60)
            # _input_threads = []
            # for t in self.input_threads:
            #     if t.is_alive():
            #         _input_threads.append(t)
            #
            #     else:
            #         new_t = Thread(target=self.fill_input_queue)
            #         _input_threads.append(new_t)
            #         _input_threads[-1].daemon = True
            #         _input_threads[-1].start()
            # self.input_threads = _input_threads

            _bucketing_threads = []
            for t in self.bucketing_threads:
                if t.is_alive():
                    _bucketing_threads.append(t)

                else:
                    new_t = Thread(target=self.fill_bucket_into_queue)
                    _bucketing_threads.append(new_t)
                    _bucketing_threads[-1].daemon = True
                    _bucketing_threads[-1].start()
            self.bucketing_threads = _bucketing_threads

if __name__ == '__main__':
    model_config = Seq2SeqAttentionModelConfig()
    data_config = Seq2SeqAttentionDataConfig()
    data_loader = DataLoader(data_config)
    batch_reader = BatchReader(model_config, data_config, data_loader)
    for i in range(12):
        result = batch_reader.next_batch()
        print(result[1])