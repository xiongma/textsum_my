
import os
import time

import tensorflow as tf

class BSDecoder(object):
    """
    beam search decoder
    """
    def __init__(self, model, model_config, vocab):
        self.model = model
        self.model_config = model_config
        self.vocab = vocab

        self.max_decode_steps = 100000
        self.decode_loop_delay_secs = 60

        self.saver = tf.train.Saver()

    def decode_loop(self):
        """
        Decoding loop for long running process
        :return:
        """
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        step = 0
        while step < self.max_decode_steps:
            time.sleep(self.decode_loop_delay_secs)

            if not self.decode(self.saver, sess):
                continue
            step += 1

    def decode(self, saver, sess):
        """
        restore a checkpoint and decode it.
        :param saver: tensorflow saver
        :param sess: tensorflow session
        :return: If success, returns true, otherwise, false.
        """
        ckpt_state = tf.train.get_checkpoint_state(self.model_config.log_path)
        if not(ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to decode yet at %s', self.model_config.log_path)
            return False

        tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
        ckpt_path = os.path.join(
            self.model_config.log_path, os.path.basename(ckpt_state.model_checkpoint_path)
        )
        tf.logging.info('renamed checkpoint path %s', ckpt_path)
        saver.restore(sess, ckpt_path)

