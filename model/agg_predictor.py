import json
import torch
import numpy as np
from SQLNet.utils import run_lstm, col_name_encode

import tensorflow as tf
import tensorflow.keras.layers as layers


class AggPredictor(tf.keras.Model):
    def __init__(self, N_word, N_h, N_depth, use_ca):
        super(AggPredictor, self).__init__()

        self.use_ca = use_ca

        self.agg_lstm = tf.keras.Sequential([layers.Bidirectional(layers.LSTM(N_h / 2,
                                                                              dropout=0.3)) for _ in range(N_depth)])
        if use_ca:
            print("Using column attention on aggregator predicting")
            self.agg_col_name_enc = tf.keras.Sequential([layers.Bidirectional(layers.LSTM(N_h / 2,
                                                                                          dropout=0.3)) for _ in
                                                         range(N_depth)])
            self.agg_att = layers.Dense(N_h)
        else:
            print("Not using column attention on aggregator predicting")
            self.agg_att = layers.Dense(1)
        self.agg_out = tf.keras.Sequential([layers.Dense(N_h, activation='tanh'), layers.Dense(6)])
        self.softmax = layers.Softmax()

    def call(self, inputs, training=None, mask=None, col_inp_var=None, col_name_len=None,
             col_len=None, col_num=None, gt_sel=None):
        x_emb_var, x_len = inputs

        max_x_len = max(x_len)

        h_enc, _ = run_lstm(self.agg_lstm, x_emb_var, x_len)
        if self.use_ca:
            e_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.agg_col_name_enc)

            chosen_sel_idx = tf.convert_to_tensor(gt_sel, dtype=tf.int64)
            aux_range = tf.convert_to_tensor(range(len(gt_sel)), dtype=tf.int64)

            chosen_e_col = e_col[aux_range, chosen_sel_idx]
            att_val = tf.squeeze(tf.matmul(self.agg_att(h_enc), tf.expand_dims(chosen_e_col, axis=2)))

        else:
            att_val = tf.squeeze(self.agg_att(h_enc))

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val[idx, num:] = -100

        att = self.softmax(att_val)

        k_agg = tf.reduce_sum((h_enc * tf.repeat(tf.expand_dims(att, axis=2), h_enc)), axis=1)
        agg_score = self.agg_out(k_agg)
        return agg_score
