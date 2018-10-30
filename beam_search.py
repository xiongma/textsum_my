
import numpy as np

class Hypothesis(object):
    """
        Defines a hypothesis during beam search.
    """
    def __init__(self, tokens, log_prob, state):
        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state

    def extend(self, token, log_prob, new_state):
        """
        Extend the hypothesis with result from latest step.
        :param token: latest token from decoding
        :param log_prob: log prob of the latest decoded tokens.
        :param new_state: decoder output state. Fed to the decoder for next step.
        :return: new Hypothesis with the results from latest step.
        """

        return Hypothesis(self.tokens + [token], self.log_prob + log_prob,
                          new_state)

    @property
    def latest_token(self):
        return self.tokens[-1]

    def __str__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob,
                                                              self.tokens))

class BeamSearch(object):
    def __init__(self, model, beam_size, start_token, end_token, max_steps):
        self.model = model
        self.beam_size = beam_size
        self.start_token = start_token
        self.end_token = end_token
        self.max_steps = max_steps

        # This length normalization is only effective for the final results.
        self.normalize_by_length = True

    def search(self, sess, enc_inputs, enc_sequence_length):
        """
        use beam search for decoding
        :param sess: tensorflow session
        :param enc_inputs: numpy array, [batch_size, words_id], which is input article
        :param enc_sequence_length: [batch_size]
        :return: hyps: list of Hypothesis, the best hypotheses found by beam search,
                       ordered by score
        """

        """
            get encoder output
        """
        encoder_top_states, decoder_in_state = sess.run([self.model.encoder_top_states, self.model.forward_state],
                                                     feed_dict={self.model.article: enc_inputs,
                                                                self.model.article_length: enc_sequence_length})

        """
            create a list, which each element is Hypothesis
        """
        hyps = [Hypothesis([self.start_token], 0.0, decoder_in_state)
                ] * self.beam_size

        results = []

        steps = 0
        while steps < self.max_steps and len(results) < self.beam_size:
            latest_tokens = [h.latest_token for h in hyps]
            states = [h.state for h in hyps]

            """
                get ids, probabilities, _states, when input last decoder output, now decoder timestep is one
            """
            ids, probs, _states = sess.run([self.model.topk_ids, self.model.topk_log_probs, self.model.decoder_outputs_state],
                     feed_dict={
                         self.model.encoder_top_states: encoder_top_states,
                         self.model.decoder_in_state: states,
                         self.model.abstact: np.transpose(np.array([latest_tokens]))
                     })
            new_states = [s for s in _states]

            all_hyps = []

            num_beam_source = 1 if steps == 0 else len(hyps)

            """
                这里的含义就是从上一次的结果中继续拼接这一次的输出，然后进行排序，降序，如果遇到结束符，这把这个结果拼接到结果中，然后降序，
                输出最大的一个
            """
            for i in range(num_beam_source):
                h, ns = hyps[i], new_states[i]
                for j in range(self.beam_size*2):
                    """
                        遍历所有可能，因为有8个假设，每个假设下面有16，那么就是128，再取前8个
                        相当于K = 8， new_k=16
                    """
                    all_hyps.append(h.extend(ids[i, j], probs[i, j], ns))

            # Filter and collect any hypotheses that have the end token
            hyps = []


            for h in self.best_hyps(all_hyps):
                if h.latest_token == self.end_token:
                    # Pull the hypothesis off the beam if the end token is reached.
                    results.append(h)
                else:
                    # Otherwise continue to the extend the hypothesis.
                    hyps.append(h)
                if len(hyps) == self.beam_size or len(results) == self.beam_size:
                    break

            steps += 1

            if steps == self.max_steps:
                results.extend(hyps)

            return self.best_hyps(results)

    def best_hyps(self, hyps):
        """
        Sort the hyps based on log probs and length.
        :param hyps: A list of hypothesis
        :return: A list of sorted hypothesis in reverse log_prob order.
        """
        # This length normalization is only effective for the final results.
        if self.normalize_by_length:
            return sorted(hyps, key=lambda h: h.log_prob / len(h.tokens), reverse=True)
        else:
            return sorted(hyps, key=lambda h: h.log_prob, reverse=True)
