import json
import numpy as np
from SQLNet.utils import run_lstm, col_name_encode

import tensorflow as tf
import tensorflow.keras.layers as layers


class SelPredictor(tf.keras.Model):
    def __init__(self, N_word, N_h, N_depth, max_tok_num, use_ca):
        super(SelPredictor, self).__init__()
        self.use_ca = use_ca
        self.max_tok_num = max_tok_num
        self.sel_lstm = tf.keras.Sequential([layers.Bidirectional(layers.LSTM(N_h / 2, dropout=.3,
                                                                              return_sequences=True)) for _ in range(
            N_depth)])
        if use_ca:
            print("Using column attention on selection predicting")
            self.sel_att = layers.Dense(N_h)
        else:
            print("Not usng column attention on selection predicting")
            self.sel_att = layers.Dense(1)

        self.sel_col_name_enc = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(N_h / 2, dropout=.3, return_sequences=True)) for _ in range(N_depth)
        ])

        self.sel_out_K = layers.Dense(N_h)
        self.sel_out_col = layers.Dense(N_h)
        self.sel_out = tf.keras.Sequential([
            layers.Activation("tanh"),
            layers.Dense(1)
        ])
        self.softmax = layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num = inputs
        B = len(x_emb_var)
        max_x_len = max(x_len)

        e_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.sel_col_name_enc)

        if self.use_ca:
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            att_val = tf.matmul(e_col, self.sel_att(h_enc), transpose_b=True)
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, :, num:] = -100
            att = tf.reshape(self.softmax(tf.reshape(att_val, (-1, max_x_len))), (B, -1, max_x_len))
            K_sel_expand = tf.reduce_sum((tf.expand_dims(h_enc, axis=1) * tf.expand_dims(att, axis=3)), axis=2)

        else:
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            att_val = tf.squeeze(self.sel_att(h_enc))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, num:] = -100

            att = self.softmax(att_val)
            K_sel = tf.reduce_sum((h_enc * tf.repeat(tf.expand_dims(att, axis=2), h_enc.shape[-1])), axis=1)
            K_sel_expand = tf.expand_dims(K_sel)

        sel_score = tf.squeeze(self.sel_out(self.sel_out_K(K_sel_expand) + self.sel_out_col(e_col)))
        max_col_num = max(col_num)
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                sel_score[idx, num:] = -100

        return sel_score
