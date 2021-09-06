import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from SQLNet.utils import run_lstm, col_name_encode


class SQLNetCondPred(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, use_ca):
        super(SQLNetCondPred, self).__init__()
        self.N_h = N_h
        self.max_tok_num = max_tok_num
        self.max_col_num = max_col_num
        self.use_ca = use_ca

        self.cond_num_lstm = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(N_h / 2, dropout=.3, return_sequences=True))
            for _ in range(N_depth)
        ])

        self.cond_num_att = layers.Dense(1)
        self.cond_num_out = tf.keras.Sequential([
            layers.Dense(N_h, activation='tanh'),
            layers.Dense(5)
        ])

        self.cond_num_name_enc = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(N_h / 2, dropout=.3, return_sequences=True))
            for _ in range(N_depth)
        ])

        self.cond_num_col_att = layers.Dense(1)
        self.cond_num_col2_hid1 = layers.Dense(2 * N_h)
        self.cond_num_col2hid2 = layers.Dense(2 * N_h)

        self.cond_col_lstm = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(N_h / 2, dropout=.3, return_sequences=True))
            for _ in range(N_depth)
        ])

        if use_ca:
            print("Using column attention on where predicting")
            self.cond_col_att = layers.Dense(N_h)
        else:
            print("Not using column attention on where predicting")
            self.cond_col_att = layers.Dense(1)

        self.cond_col_name_enc = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(N_h / 2, dropout=.3, return_sequences=True))
            for _ in range(N_depth)
        ])

        self.cond_col_out_K = layers.Dense(N_h)
        self.cond_col_out_col = layers.Dense(N_h)
        self.cond_col_out = tf.keras.Sequential([
            layers.Activation("relu"),
            layers.Dense(1)
        ])

        self.cond_op_lstm = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(N_h / 2, dropout=.3, return_sequences=True))
            for _ in range(N_depth)
        ])

        if use_ca:
            self.cond_op_att = layers.Dense(N_h)
        else:
            self.cond_op_att = layers.Dense(1)

        self.cond_op_out_K = layers.Dense(N_h)
        self.cond_op_name_enc = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(N_h / 2, dropout=.3, return_sequences=True))
            for _ in range(N_depth)
        ])
        self.cond_op_out_col = layers.Dense(N_h)
        self.cond_op_out = tf.keras.Sequential([
            layers.Dense(N_h, activation='tanh'),
            layers.Dense(3)
        ])

        self.cond_str_lstm = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(N_h / 2, dropout=.3, return_sequences=True))
            for _ in range(N_depth)
        ])

        self.cond_str_decoder = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(N_h, dropout=.3, return_sequences=True))
            for _ in range(N_depth)
        ])

        self.cond_str_name_enc = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(N_h / 2, dropout=.3, return_sequences=True))
            for _ in range(N_depth)
        ])

        self.cond_str_out_g = layers.Dense(N_h)
        self.cond_str_out_h = layers.Dense(N_h)
        self.cond_str_out_col = layers.Dense(N_h)
        self.cond_str_out = tf.keras.Sequential([
            layers.Activation("relu"),
            layers.Dense(1)
        ])

        self.softmax = layers.Softmax()

    def gen_gt_batch(self, split_tok_seq):
        B = len(split_tok_seq)
        max_len = max([max([len(tok) for tok in tok_seq] + [0]) for
                       tok_seq in split_tok_seq]) - 1  # The max seq len in the batch.
        if max_len < 1:
            max_len = 1
        ret_array = np.zeros((
            B, 4, max_len, self.max_tok_num), dtype=np.float32)
        ret_len = np.zeros((B, 4))
        for b, tok_seq in enumerate(split_tok_seq):
            idx = 0
            for idx, one_tok_seq in enumerate(tok_seq):
                out_one_tok_seq = one_tok_seq[:-1]
                ret_len[b, idx] = len(out_one_tok_seq)
                for t, tok_id in enumerate(out_one_tok_seq):
                    ret_array[b, idx, t, tok_id] = 1
            if idx < 3:
                ret_array[b, idx + 1:, 0, 1] = 1
                ret_len[b, idx + 1:] = 1

        ret_inp = tf.convert_to_tensor(ret_array)
        if self.gpu:
            ret_inp = ret_inp.cuda()
        ret_inp_var = tf.Variable(ret_inp)

        return ret_inp_var, ret_len  # [B, IDX, max_len, max_tok_num]

    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len,
                col_len, col_num, gt_where, gt_cond, reinforce):
        max_x_len = max(x_len)
        B = len(x_len)
        if reinforce:
            raise NotImplementedError('Our model doesn\'t have RL')

        # Predict the number of conditions
        # First use column embeddings to calculate the initial hidden unit
        # Then run the LSTM and predict condition number.
        e_num_col, col_num = col_name_encode(col_inp_var, col_name_len,
                                             col_len, self.cond_num_name_enc)
        num_col_att_val = tf.squeeze(self.cond_num_col_att(e_num_col))
        for idx, num in enumerate(col_num):
            if num < max(col_num):
                num_col_att_val[idx, num:] = -100
        num_col_att = self.softmax(num_col_att_val)
        K_num_col = tf.reduce_sum(e_num_col * tf.expand_dims(num_col_att, axis=2), axis=1)
        cond_num_h1 = tf.transpose(tf.reshape(self.cond_num_col2hid1(K_num_col), shape=(B, 4, self.N_h / 2)),
                                   perm=(1, 0, 2))
        cond_num_h2 = tf.transpose(tf.reshape(self.cond_num_col2hid2(K_num_col), shape=(B, 4, self.N_h / 2)),
                                   perm=(1, 0, 2))

        h_num_enc, _ = run_lstm(self.cond_num_lstm, x_emb_var, x_len,
                                hidden=(cond_num_h1, cond_num_h2))

        num_att_val = tf.squeeze(self.cond_num_att(h_num_enc))

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                num_att_val[idx, num:] = -100
        num_att = self.softmax(num_att_val)

        K_cond_num = tf.reduce_sum(h_num_enc * tf.repeat(tf.expand_dims(num_att, axis=2), h_num_enc.shape[
            -1]), axis=1)
        cond_num_score = self.cond_num_out(K_cond_num)

        # Predict the columns of conditions
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len, col_len,
                                        self.cond_col_name_enc)

        h_col_enc, _ = run_lstm(self.cond_col_lstm, x_emb_var, x_len)
        if self.use_ca:
            col_att_val = tf.matmul(e_cond_col,
                                    tf.transpose(self.cond_col_att(h_col_enc), perm=(0, 2, 1)))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    col_att_val[idx, :, num:] = -100
            col_att = tf.reshape(self.softmax(tf.reshape(col_att_val, shape=(-1, max_x_len))), shape=(B, -1, max_x_len))
            K_cond_col = tf.reduce_sum(tf.expand_dims(h_col_enc, axis=1) * tf.expand_dims(col_att, axis=3), axis=2)
        else:
            col_att_val = tf.squeeze(self.cond_col_att(h_col_enc))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    col_att_val[idx, num:] = -100
            col_att = self.softmax(col_att_val)
            K_cond_col = tf.expand_dims(tf.reduce_sum(h_col_enc *
                                                      tf.expand_dims(col_att_val, axis=2), axis=1), axis=1)

        cond_col_score = tf.squeeze(self.cond_col_out(self.cond_col_out_K(K_cond_col) +
                                                      self.cond_col_out_col(e_cond_col)))
        max_col_num = max(col_num)
        for b, num in enumerate(col_num):
            if num < max_col_num:
                cond_col_score[b, num:] = -100

        # Predict the operator of conditions
        chosen_col_gt = []
        if gt_cond is None:
            cond_nums = np.argmax(cond_num_score.numpy(), axis=1)
            col_scores = cond_col_score.numpy()
            chosen_col_gt = [list(np.argsort(-col_scores[b])[:cond_nums[b]])
                             for b in range(len(cond_nums))]
        else:
            chosen_col_gt = [[x[0] for x in one_gt_cond] for
                             one_gt_cond in gt_cond]

        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len,
                                        col_len, self.cond_op_name_enc)
        col_emb = []
        for b in range(B):
            cur_col_emb = tf.stack([e_cond_col[b, x]
                                    for x in chosen_col_gt[b]] + [e_cond_col[b, 0]] *
                                   (4 - len(chosen_col_gt[b])))  # Pad the columns to maximum (4)
            col_emb.append(cur_col_emb)
        col_emb = tf.stack(col_emb)

        h_op_enc, _ = run_lstm(self.cond_op_lstm, x_emb_var, x_len)
        if self.use_ca:
            op_att_val = tf.squeeze(tf.matmul(tf.expand_dims(self.cond_op_att(h_op_enc), axis=1),
                                              tf.expand_dims(col_emb, axis=3)))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    op_att_val[idx, :, num:] = -100
            op_att = tf.reshape(self.softmax(tf.reshape(op_att_val, shape=(B * 4, -1))), shape=(B, 4, -1))
            K_cond_op = tf.reduce_sum(tf.expand_dims(h_op_enc, axis=1) * tf.expand_dims(op_att, axis=3), axis=2)
        else:
            op_att_val = tf.squeeze(self.cond_op_att(h_op_enc))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    op_att_val[idx, num:] = -100
            op_att = self.softmax(op_att_val)
            K_cond_op = tf.expand_dims(tf.reduce_sum(h_op_enc * tf.expand_dims(op_att, axis=2), axis=1), axis=1)

        cond_op_score = tf.squeeze(self.cond_op_out(self.cond_op_out_K(K_cond_op) +
                                                    self.cond_op_out_col(col_emb)))

        # Predict the string of conditions
        h_str_enc, _ = run_lstm(self.cond_str_lstm, x_emb_var, x_len)
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len,
                                        col_len, self.cond_str_name_enc)
        col_emb = []
        for b in range(B):
            cur_col_emb = tf.stack([e_cond_col[b, x]
                                    for x in chosen_col_gt[b]] +
                                   [e_cond_col[b, 0]] * (4 - len(chosen_col_gt[b])))
            col_emb.append(cur_col_emb)
        col_emb = tf.stack(col_emb)

        if gt_where is not None:
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where)
            g_str_s_flat, _ = tf.reshape(self.cond_str_decoder(
                gt_tok_seq, shape=(B * 4, -1, self.max_tok_num)))
            g_str_s = tf.reshape(g_str_s_flat, shape=(B, 4, -1, self.N_h))

            h_ext = tf.expand_dims(tf.expand_dims(h_str_enc, axis=1), axis=1)
            g_ext = tf.expand_dims(g_str_s, axis=3)
            col_ext = tf.expand_dims(tf.expand_dims(col_emb, axis=2), axis=2)

            cond_str_score = tf.squeeze(self.cond_str_out(
                self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext) +
                self.cond_str_out_col(col_ext)))
            for b, num in enumerate(x_len):
                if num < max_x_len:
                    cond_str_score[b, :, :, num:] = -100
        else:
            h_ext = tf.expand_dims(tf.expand_dims(h_str_enc, axis=1), axis=1)
            col_ext = tf.expand_dims(tf.expand_dims(col_emb, axis=2), axis=2)
            scores = []

            t = 0
            init_inp = np.zeros((B * 4, 1, self.max_tok_num), dtype=np.float32)
            init_inp[:, 0, 0] = 1  # Set the <BEG> token
            cur_inp = tf.Variable(tf.convert_to_tensor(init_inp))
            cur_h = None
            while t < 50:
                if cur_h:
                    g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp, cur_h)
                else:
                    g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp)
                g_str_s = tf.reshape(g_str_s_flat, shape=(B, 4, 1, self.N_h))
                g_ext = tf.expand_dims(g_str_s, axis=3)

                cur_cond_str_score = tf.squeeze(self.cond_str_out(
                    self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext)
                    + self.cond_str_out_col(col_ext)))
                for b, num in enumerate(x_len):
                    if num < max_x_len:
                        cur_cond_str_score[b, :, num:] = -100
                scores.append(cur_cond_str_score)

                ans_tok_var = tf.reduce_max(cur_cond_str_score.view(B * 4, max_x_len), axis=1)
                ans_tok = ans_tok_var
                data = tf.one_hot(tf.expand_dims(ans_tok, axis=1), self.max_tok_num)
                cur_inp = tf.Variable(data)
                cur_inp = tf.expand_dims(cur_inp, axis=1)

                t += 1

            cond_str_score = tf.stack(scores, 2)
            for b, num in enumerate(x_len):
                if num < max_x_len:
                    cond_str_score[b, :, :, num:] = -100  # [B, IDX, T, TOK_NUM]

        cond_score = (cond_num_score,
                      cond_col_score, cond_op_score, cond_str_score)

        return cond_score

    def call(self, inputs, training=None, mask=None):
        return self.forward(*inputs)
