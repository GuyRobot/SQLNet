import json
import numpy as np
from SQLNet.utils import run_lstm

import tensorflow as tf
import tensorflow.keras.layers as layers


class Seq2SQLCondPred(tf.keras.Model):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num):
        super(Seq2SQLCondPred, self).__init__()
        self.N_h = N_h
        self.max_tok_num = max_tok_num
        self.max_col_num = max_col_num

        self.cond_lstm = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(N_h / 2, dropout=.3, return_sequences=True)) for _ in range(N_depth)
        ])
        self.cond_decoder = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(N_h, dropout=3, return_sequences=True)) for _ in range(N_depth)
        ])

        self.cond_out_g = layers.Dense(N_h)
        self.cond_out_h = layers.Dense(N_h)
        self.cond_out = tf.keras.Sequential([
            layers.Activation('tanh'),
            layers.Dense(1)
        ])

        self.softmax = layers.Softmax()

    def gen_gt_batch(self, tok_seq, gen_inp=True):
        # If gen_inp: generate the input token sequence (removing <END>)
        # Otherwise: generate the output token sequence (removing <BEG>)
        B = len(tok_seq)
        ret_len = np.array([len(one_tok_seq) - 1 for one_tok_seq in tok_seq])
        max_len = max(ret_len)
        ret_array = np.zeros([B, max_len, self.max_tok_num], dtype=np.float32)
        for b, one_tok_seq in enumerate(tok_seq):
            out_one_tok_seq = one_tok_seq[:-1] if gen_inp else one_tok_seq[1:]
            for i, tok_id in enumerate(out_one_tok_seq):
                ret_array[b, i, tok_id] = 1

        ret_inp = tf.convert_to_tensor(ret_array)
        ret_inp_var = tf.Variable(ret_inp)

        return ret_inp_var, ret_len

    def call(self, inputs, training=None, mask=None):
        x_emb_var, x_len, col_inp_var, col_name_len, col_len, \
        col_num, gt_where, gt_cond, reinforce = inputs

        max_x_len = max(x_len)
        B = len(x_len)

        h_enc, hidden = run_lstm(self.cond_lstm, x_emb_var, x_len)
        decoder_hidden = tuple(tf.concat((hid[:2], hid[2:]), axis=2) for hid in hidden)

        if gt_where is not None:
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where, gen_inp=True)
            g_s, _ = run_lstm(self.cond_decoder, gt_tok_seq, gt_tok_len, decoder_hidden)

            h_enc_expand = tf.expand_dims(h_enc, axis=1)
            g_s_expand = tf.expand_dims(g_s, axis=2)
            cond_score = tf.squeeze(self.cond_out(self.cond_out_h(h_enc_expand) + self.cond_out_g(g_s_expand)))

            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    cond_score[idx, :, num:] = -100
        else:
            h_enc_expand = tf.expand_dims(h_enc, axis=1)
            scores = []
            choices = []
            done_set = set()

            t = 0
            init_inp = np.zeros((B, 1, self.max_tok_num), dtype=np.float32)
            init_inp[:, 0.7] = 1  # BEG token

            cur_inp = tf.Variable(tf.convert_to_tensor(init_inp))
            cur_h = decoder_hidden
            while len(done_set) < B and t < 100:
                g_s, cur_h = self.cond_decoder(cur_inp, cur_h)
                g_s_expand = tf.expand_dims(g_s, axis=2)

                cur_cond_score = tf.squeeze(self.cond_out(self.cond_out_h(h_enc_expand) + self.cond_out_g(g_s_expand)))

                for b, num in enumerate(x_len):
                    if num < max_x_len:
                        cur_cond_score[b, num:] = -100
                scores.append(cur_cond_score)

                if not reinforce:
                    ans_tok_var = tf.reduce_max(tf.reshape(cur_cond_score, shape=(B, max_x_len)), axis=1)
                    ans_tok_var = tf.expand_dims(ans_tok_var, axis=1)
                else:
                    ans_tok_var = tf.random.categorical(self.softmax(cur_cond_score))
                    choices.append(ans_tok_var)

                cur_inp = tf.Variable(tf.one_hot(ans_tok_var, self.max_tok_num))
                cur_inp = tf.expand_dims(cur_inp, axis=1)

                for idx, tok in enumerate(tf.squeeze(ans_tok_var)):
                    if tok == 1:  # END token
                        done_set.add(idx)

                t += 1

            cond_score = tf.stack(scores, axis=1)

        if reinforce:
            return cond_score, choices
        else:
            return cond_score