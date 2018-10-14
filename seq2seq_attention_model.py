# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn

import seq2seq_lib

class Seq2SeqAttentionModel(object):
    def __init__(self, model_config, embedding, model_is_decode=False):
        self.model_config = model_config
        self.embedding = embedding
        self.words_dict_len = len(self.embedding) + 1

        self.add_placeholder()
        self.init_weight()

        self.inference()
        self.encoder_outputs, self.forward_state, _ = self.encoder(self.article_emb_transpose)
        self.decoder_outputs, self.decoder_outputs_state = self.decoder(self.encoder_outputs, self.forward_state)
        self.outputs = self.output(self.decoder_outputs, model_is_decode)

        if not model_is_decode:
            self.loss = self.calculate_loss(self.decoder_outputs, self.outputs[0])
            self.optim = self.textsum_train(self.loss)

    def inference(self):
        """
        this function is able to inference model
        :return:
        """
        with tf.name_scope('word_embedding'):
            """
                self.article_emb : [batch_size, article_length, embedding_size]
                self.abstract_emb : [batch_size, abstract_length, embedding_size]
            """
            self.article_emb = tf.nn.embedding_lookup(self.embedding, self.article)
            self.abstract_emb = tf.nn.embedding_lookup(self.embedding, self.abstract)

        with tf.name_scope('transpose'):
            """
                self.article_emb_transpose : list, which elements shape [batch_size, embedding_size]
                self.abstract_emb_transpose : list, which elements shape [batch_size, embedding_size]
            """
            self.article_transpose = tf.unstack(tf.transpose(self.article))
            self.abstract_transpose = tf.unstack(tf.transpose(self.abstract))
            
            self.article_emb_transpose = [tf.nn.embedding_lookup(self.embedding, x) for x in self.article_transpose]
            self.abstract_emb_transpose = [tf.nn.embedding_lookup(self.embedding, x) for x in self.abstract_transpose]

    def encoder(self, input_content_emb):
        """
        this function is able to encoder input content
        :param input_content_emb: input content embedding
        :return: encoder output, forward state, backward state
        """
        forward_state = None
        backward_state = None
        encoder_outputs = None

        with tf.variable_scope('encoder'):
            for layer_i in range(self.model_config.encoder_layer_num):
                with tf.variable_scope(str(layer_i)):
                    forward_cell = self.create_gru_unit(self.model_config.cell_output_size, self.model_config.cell_output_prob,
                                                        'forward')
                    backward_cell = self.create_gru_unit(self.model_config.cell_output_size, self.model_config.cell_output_prob,
                                                         'backward')

                    """
                        rnn.static_bidirectional_rnn : inputs must have shape, which shape params is 
                        [time_steps, batch_size, hidden_size]
                        outputs shape is [time_steps, batch_size, 2 * hidden_size]
                        forward_state shape is [batch_size, 2 * hidden_size]
                        backward_state shape is [batch_size, 2 * hidden_size]
                    """
                    encoder_outputs, forward_state, backward_state = rnn.static_bidirectional_rnn(forward_cell,
                                                        backward_cell, input_content_emb, dtype=tf.float32,
                                                        sequence_length=1)
        """
            encoder_outputs : [article_length/time_steps, batch_size, 2 * cell_output_size]
            forward_state : [batch_size, 2 * cell_output_size], last forward state add first backward state
            backward_state : [batch_size, 2 * cell_output_size] last backward state add first forward state
        """
        return encoder_outputs, forward_state, backward_state

    def decoder(self, encoder_outputs, forward_state):
        """
        this function is able to decoder encoder input
        :param encoder_outputs: encoder output
        :param forward_state: forward cell state in the end
        :return:
        """
        with tf.variable_scope('decoder'), tf.name_scope('decoder'):
            loop_function = self._extract_argmax_and_embed((self.w, self.v), update_embedding=False)
            """
                encoder_outputs_ is list, which each element shape is [batch_size, 1, 2 * cell_output_size]
            """
            encoder_outputs_ = [tf.reshape(x, [self.model_config.batch_size, 1, 2 * self.model_config.cell_output_size])
                               for x in encoder_outputs]
            """
                encoder_outputs_ shape is [batch_size, article_length, 2 * cell_output_size]
            """
            encoder_top_states = tf.concat(axis=1, values=encoder_outputs_)

            with tf.variable_scope('attention'), tf.name_scope('attention'):
                cell = self.create_gru_unit(self.model_config.cell_output_size, self.model_config.cell_output_prob,
                                            name_scope='decoder')

                """
                    decoder_outputs : [summary_length， batch_size, hidden_size]
                """
                decoder_outputs, decoder_outputs_state = tf.contrib.legacy_seq2seq.attention_decoder(self.article_emb,
                                                            forward_state, encoder_top_states, cell, num_heads=1,
                                                            loop_function=loop_function, initial_state_attention=True)

        return decoder_outputs, decoder_outputs_state

    def output(self, decoder_outputs, model_is_decode=False):
        """
        calculate outputs, if op is decode, return decode_output
        :param decoder_outputs: decoder outputs
        :param model_is_decode: whether current op is decode, Default False
        :return:
        """
        with tf.variable_scope('output'), tf.name_scope('output'):
            model_outputs = []
            for i in range(len(decoder_outputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                    """
                        this is able to transfer cell_output_size to embedding vocab size, use linear function
                        this is call soft alignment
                    """
                    model_outputs.append(tf.nn.xw_plus_b(decoder_outputs[i], self.w, self.v))

        if not model_is_decode:
            return model_outputs, None, None

        else:
            with tf.variable_scope('decoder_output'), tf.name_scope('decoder_output'):
                """
                    model_outputs : [time_steps, batch_size, vocab_size]
                    best_outputs : [time_steps, batch_size]
                    this is get position of vocab
                """
                best_outputs = [tf.argmax(x, 1) for x in model_outputs]

                """
                    summarise_ids : [batch_size, time_steps]
                    this is output summarise, time steps is decoder time steps, in each time steps elements is vocab id
                """
                summarise_ids = tf.concat(axis=1, values=[tf.reshape(x, [self.model_config.batch_size, 1])
                                                          for x in best_outputs])
                """
                    output last time step top k, it's call summary id
                """
                topk_log_probs, topk_ids = tf.nn.top_k(
                    tf.log(tf.nn.softmax(model_outputs[-1])), self.model_config.batch_size * 2)

            return summarise_ids, topk_log_probs, topk_ids

    def calculate_loss(self, decoder_outputs, model_outputs):
        """
        calculate loss
        :param decoder_outputs: decoder outputs
        :param model_outputs: soft alignment
        :return: loss
        """
        with tf.variable_scope('loss'), tf.name_scope('loss'):
            def sampled_loss_func(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(
                    weights=self.w_t, biases=self.v, labels=labels, inputs=inputs,
                    num_sampled=self.model_config.num_softmax_samples, num_classes=self.words_dict_len)

            if self.model_config.num_softmax_samples != 0 and self.model_config.model == 'train':
                loss = seq2seq_lib.sampled_sequence_loss(
                                                    decoder_outputs, self.targets, self.loss_weights, sampled_loss_func)
            else:
                loss = tf.contrib.legacy_seq2seq.sequence_loss(
                    model_outputs, self.targets, self.loss_weights)

        return loss

    def textsum_train(self, loss):
        """
        SGD loss
        :param loss: loss
        :return:
        """
        with tf.name_scope('train'):
            optim = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

            return optim

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

    def add_placeholder(self):
        """
        this function is able to add tensorflow place holder
        :return:
        """
        self.article = tf.placeholder(tf.int32, [self.model_config.batch_size, self.model_config.article_length], name='article')

        self.abstract = tf.placeholder(tf.int32, [self.model_config.batch_size, self.model_config.abstract_length], name='abstract')

        self.targets = tf.placeholder(tf.int32, [self.model_config.batch_size, self.model_config.abstract_length], name='targets')

        self.loss_weights = tf.placeholder(tf.float32, [self.model_config.batch_size, self.model_config.abstract_length],
                                           name='loss_weights')

        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    def init_weight(self):
        """
        this function is able to init network weight
        :return:
        """
        with tf.variable_scope('output_projection'):
            self.w = tf.get_variable('w', [self.model_config.cell_output_size, self.words_dict_len], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=1e-4))
            self.w_t = tf.transpose(self.w)

            self. v = tf.get_variable('v', [self.words_dict_len], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1e-4))