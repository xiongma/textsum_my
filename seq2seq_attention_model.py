# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn

class Seq2SeqAttentionModel(object):
    def __init__(self, config, embedding):
        self.config = config
        self.embedding = embedding

        self.add_placeholder()
        self.init_weight()

        self.inference()
        encoder_output, forward_state, _ = self.encoder(self.input_content_emb)

    def inference(self):
        """
        this function is able to inference model
        :return:
        """
        with tf.name_scope('word_embedding'):
            self.input_content_emb = tf.nn.embedding_lookup(self.embedding, self.input_content)

    def encoder(self, input_content_emb):
        """
        this function is able to encoder input content
        :param input_content_emb: input content embedding
        :return: encoder output, forward state, backward state
        """
        forward_state = None
        backward_state = None
        encoder_output = None

        with tf.variable_scope('encoder'):
            for layer_i in range(self.config.encoder_layer_num):
                with tf.variable_scope(str(layer_i)):
                    forward_cell = self.create_gru_unit(self.config.cell_output_size, self.config.cell_output_prob,
                                                        'forward')
                    backward_cell = self.create_gru_unit(self.config.cell_output_size, self.config.cell_output_prob,
                                                         'backward')
                    """
                        encoder : [batch_size, sequence_length, 2 * cell_output_size]
                        forward_state : [batch_size, 2 * cell_output_size]
                        backward_state : [batch_size, 2 * cell_output_size]
                    """
                    encoder_output, forward_state, backward_state = rnn.static_bidirectional_rnn(forward_cell,
                                                        backward_cell, input_content_emb, dtype=tf.float32,
                                                        sequence_length=self.config.sequence_length)

        return encoder_output, forward_state, backward_state

    def decoder(self, encoder_output, forward_state):
        """
        this function is able to decoder encoder input
        :param encoder_output: encoder output
        :param forward_state: forward cell state in the end
        :return:
        """
        with tf.variable_scope('decoder'), tf.name_scope('decoder'):
            loop_function = self._extract_argmax_and_embed((self.w, self.v), update_embedding=False)

            with tf.variable_scope('attention'), tf.name_scope('attention'):
                cell = self.create_gru_unit(self.config.cell_output_size, self.config.cell_output_prob, name_scope='decoder')

                """
                    list : each value is [batch_size, 2 * cell_output_size], length is sequence length
                """
                # encoder_output = tf.unstack(encoder_output, axis=1)
                """
                    decoder_outputs : [batch_size, summary_length, hidden_size]
                """
                decoder_outputs, decoder_output_state = tf.contrib.legacy_seq2seq.attention_decoder(self.input_content_emb,
                                                            forward_state, encoder_output, cell, num_heads=1,
                                                            loop_function=loop_function, initial_state_attention=True)

            with tf.variable_scope('output'), tf.name_scope('output'):
                model_outputs = []
                for i in range(len(decoder_outputs)):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                        model_outputs.append(tf.nn.xw_plus_b(decoder_outputs[i], self.w, self.v))

                    best_outputs = [tf.argmax(x, 1) for x in model_outputs]


    def add_placeholder(self):
        """
        this function is able to add tensorflow place holder
        :return:
        """
        self.input_content = tf.placeholder(tf.int32, [self.config.batch_size, self.config.sequence_length],
                                            name='input_content')

    def create_gru_unit(self, gru_hidden_size, gru_output_keep_prob, name_scope=None):
        """
        create gru unit
        :param gru_hidden_size: GRU output hidden_size
        :param gru_output_keep_prob: GRU output keep probability
        :param name_scope: GRU name scope
        :return: GRU cell
        """
        with tf.name_scope(name_scope):
            gru_cell = rnn.GRUCell(gru_hidden_size)
            gru_cell = rnn.DropoutWrapper(cell=gru_cell, input_keep_prob=1.0,
                                          output_keep_prob=gru_output_keep_prob)

        return gru_cell

    def _extract_argmax_and_embed(self, output_projection=None, update_embedding=True):
        """
        get a loop_function that extracts the previous symbol and embeds it
        :param output_projection: None or a pair (W, B). If provided, each fed previous
                                  output will first be multiplied by W and added B
        :param update_embedding: Boolean; if False, the gradients will not propagate
                                 through the embeddings.
        :return: a loop function
        """
        def loop_function(prev, _):
            """
            function that feed previous model output rather than ground truth.
            :param prev:
            :return:
            """
            if output_projection is not None:
                prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])

            prev_symbol = tf.argmax(prev, 1)

            emb_prev = tf.nn.embedding_lookup(self.embedding, prev_symbol)
            if not update_embedding:
                emb_prev = tf.stop_gradient(emb_prev)
            return emb_prev

        return loop_function

    def init_weight(self):
        """
        this function is able to init network weight
        :return:
        """
        with tf.variable_scope('output_projection'):
            self.w = tf.get_variable('w', [self.config.cell_output_size], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=1e-4))
            self.w_t = tf.transpose(self.w)

            self. v = tf.get_variable('v', [len(self.embedding)], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1e-4))