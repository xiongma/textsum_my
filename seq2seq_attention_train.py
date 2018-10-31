# -*- coding: utf-8 -*-

import tensorflow as tf

from seq2seq_attention_config import Seq2SeqAttentionDataConfig
from seq2seq_attention_config import Seq2SeqAttentionModelConfig
from seq2seq_attention_model import Seq2SeqAttentionModel
from batch_reader import BatchReader
from data_loader import DataLoader

class Seq2SeqAttentionTrain(object):
    def __init__(self, model_config, data_config, data_loader):
        self.model_config = model_config
        self.data_config = data_config

        self.data_loader = data_loader
        self.batch_reader = BatchReader(self.model_config, self.data_config, self.data_loader)

        self.model = Seq2SeqAttentionModel(self.model_config, self.data_loader.word2vec_vectors)

    def train(self):
        """
        train model
        :return:
        """
        """
            Train dir is different from log_root to avoid summary directory
        """
        with tf.device('/cpu:0'):
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(self.model_config.train_dir)

            sv = tf.train.Supervisor(logdir=self.model_config.log_path,
                                     is_cheif=True,
                                     saver=saver,
                                     summary_op=None,
                                     save_model_secs=self.model_config.save_model_secs,
                                     global_step=self.model.global_step)
            session = sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True))

            run_avg_loss = 0
            step = 0

            while not sv.should_stop() and step < self.model_config.max_step:
                (article_batch, abstract_batch, target_batch, article_batch_lens, dec_output_lens,
                loss_weights, _, _) = self.batch_reader.next_batch()

                to_return = [self.model.optim, self.model.summarise, self.model.loss, self.model.global_step]
                result = session.run(to_return, feed_dict={
                    self.model.article: article_batch,
                    self.model.abstract: abstract_batch,
                    self.model.targets: target_batch,
                    self.model.article_length: article_batch_lens,
                    self.model.loss_weights: loss_weights})

                avg_loss = self.running_avg_loss(result[2], run_avg_loss, summary_writer, step)

                summary_writer.add_summary(result[1], result[3])
                step += 1

                if step % 100 == 0:
                    summary_writer.flush()

                print('{0} step, loss is {1}'.format(str(step), str(avg_loss)))

            sv.stop()

    def running_avg_loss(self, loss, running_avg_loss, summary_writer, step, decay=0.999):
        """
        calculate the running average of losses.
        :param loss: model output loss
        :param running_avg_loss:
        :param summary_writer: tensorflow summary writer
        :param step: running step
        :param decay:
        :return:
        """
        if running_avg_loss == 0:
            running_avg_loss = loss
        else:
            running_avg_loss = running_avg_loss * decay + (1-decay) * running_avg_loss

        running_avg_loss = min(running_avg_loss, 12)
        loss_sum = tf.Summary()
        loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
        summary_writer.add_summary(loss_sum, step)

        return running_avg_loss

if __name__ == '__main__':
    model_config = Seq2SeqAttentionModelConfig()
    data_config = Seq2SeqAttentionDataConfig()

    data_loader = DataLoader(data_config)

    seq2seq_attention_train = Seq2SeqAttentionTrain(model_config, data_config, data_loader)

    if model_config.model == 'train':
        seq2seq_attention_train.train()

    # elif model_config.model == 'decode':

