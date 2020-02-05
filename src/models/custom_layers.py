import torch
import torch.nn as nn
import torch.nn.functional as F


class SentinelLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(SentinelLSTM, self).__init__()
        # NB! there is a difference between LSTMCell and LSTM.
        # LSTM is notably much quicker
        self.lstm_kernel = nn.LSTMCell(input_size, hidden_size)
        self.x_gate = nn.Linear(input_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, states):
        # remember old states
        h_tm1, c_tm1 = states
        # get new states
        h_t, c_t = self.lstm_kernel(x, (h_tm1, c_tm1))
        # compute sentinel vector
        # could either concat h_tm1 with x or have to gates
        sv = F.sigmoid(self.h_gate(h_tm1) + self.x_gate(x))
        s_t = sv * F.tanh(c_t)
        return h_t, c_t, s_t


class AttentionLayer(nn.Module):

    def __init__(self, input_size):
        # TODO: implement this
        super(AttentionLayer, self).__init__()
        # input_size [k, d]
        self.k = input_size[0]
        self.d = input_size[1]

        self.v_fc = nn.Linear(self.k, self.d)
        self.s_fc = nn.Linear(self.d, self.d)

        self.g_fc = nn.Linear(self.k, self.d)

        self.h_fc = nn.Linear(self.d, self.k)

    def forward(self, x):
        # TODO: implement this
        # x : [V, s_t, h_t]
        # V = [v1, v2, ..., vk]
        V = x[0]
        s_t = x[1]
        h_t = x[2]

        global_h = self.g_fc(h_t)

        mr = F.tanh(self.v_fc(V) + global_h)
        z_t = self.h_fc(mr)
        alphas = F.softmax(z_t)
