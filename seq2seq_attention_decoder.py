
import os
import tensorflow as tf

from beam_search import BeamSearch

class BSDecoder(object):
    def __init__(self, model, batch_reader, model_config, data_config, vocab, data_loader):
        self.model = model
        self.batch_reader = batch_reader
        self.model_config = model_config
        self.data_config = data_config
        self.vocab = vocab
        self.data_loader = data_loader

        self.saver = tf.train.Saver()
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        self.restore_model_flag = self.restore_model()

        self.bs = BeamSearch(self.model, self.model_config.beam_size,
                             self.data_loader.word_to_id(self.data_config.sentence_start),
                             self.data_loader.word_to_id(self.data_config.sentence_end),
                             self.model_config.abstract_length)

    def restore_model(self):
        """
        restore model
        :return: if restore model success, return True, else return False
        """
        ckpt_state = tf.train.get_checkpoint_state(self.model_config.model_path)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            print('No model to decode yet at {0}'.format(self.model_config.model_path))
            return False

        ckpt_path = os.path.join(self.model_config.model_path, os.path.basename(ckpt_state.model_checkpoint_path))

        self.saver.restore(self.session, ckpt_path)

        return True

    def decode(self, article):
        """
        decode article to abstract by model
        :param article: article, which is word id list
        :return: abstract
        """
        if self.restore_model_flag:
            """
                convert to list, which list length is beam size
            """
            article_batch = article * self.model_config.beam_size
            article_length_batch = [len(article)] * self.model_config.beam_size

            best_beam = self.bs.search(self.session, article_batch, article_length_batch)[0]

            """
                get word id after 1, because 1 is start id
            """
            result = [int(word_id) for word_id in best_beam[1:]]

            return result

        else:
            return None