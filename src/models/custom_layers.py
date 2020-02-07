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


class ImageEncoder(nn.Module):

    def __init__(self, input_shape, hidden_size, pool_size=8):
        super(ImageEncoder, self).__init__()
        self.average_pool = nn.AvgPool2d(pool_size)
        # affine transformation of attention features
        self.v_affine = nn.Linear(input_shape, hidden_size)
        # affine transformation of global image features
        self.global_affine = nn.Linear(input_shape, hidden_size)

    def forward(self, x):
        # x = V, (batch_size, 8, 8, 1536)
        input_shape = x.size()
        pixels = input_shape[1] * input_shape[2]  # 8x8 = 64
        global_image = self.average_pool(x).view(input_shape[0], -1)
        inputs = x.view(input_shape[0], pixels, input_shape[3])

        # transform
        global_image = self.global_affine(global_image)
        inputs = self.v_affine(inputs)

        return global_image, inputs


class MultimodalDecoder(nn.Module):

    def __init__(self, input_shape, hidden_size, n=0):
        super(MultimodalDecoder, self).__init__()
        self.layers = []
        if n:
            new_input_shape = input_shape + (input_shape // 2)
            self.layers.append(nn.Linear(input_shape, new_input_shape))
            input_shape = new_input_shape
        else:
            n = 1

        for _ in range(n - 1):
            self.layers.append(nn.Linear(input_shape, input_shape))

        # output layer, this is the only layer if n=0
        self.output_layer = nn.Linear(input_shape, hidden_size)

    def forward(self, x):
        # x: context vector, h_t (batch_size, 2, hidden_size)
        concat = torch.cat((x[0], x[1]))
        for layer in self.layers:
            y = F.relu(layer(concat))
            concat = torch.cat((concat, y))

        # softmax on output
        y = F.softmax(self.output_layer(concat))
        return y





class AttentionLayer(nn.Module):

    def __init__(self, input_size):
        # TODO: implement this
        super(AttentionLayer, self).__init__()
        # input_size [k, d] (64, hidden_size)
        self.k = input_size[0]
        self.d = input_size[1]

        self.v_att = nn.Linear(self.k, self.d)
        self.s_att = nn.Linear(self.d, self.d)

        self.g_att = nn.Linear(self.k, self.d)

        self.h_att = nn.Linear(self.d, self.k)

    def forward(self, x):
        # TODO: implement this
        # x : [V, s_t, h_t]
        # V = [v1, v2, ..., vk]
        # c_t is context vector: sum of alphas*v
        # output should be beta*s_t + (1-beta)*c_t

        V = x[0]
        s_t = x[1]
        h_t = x[2]

        global_h = self.g_fc(h_t)

        mr = F.tanh(self.v_fc(V) + global_h)
        z_t = self.h_fc(mr)
        alphas = F.softmax(z_t)
