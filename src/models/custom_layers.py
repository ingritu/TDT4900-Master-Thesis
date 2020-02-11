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
        print('Sentinel LSTM')
        # remember old states
        h_tm1, c_tm1 = states
        # get new states
        h_t, c_t = self.lstm_kernel(x, (h_tm1, c_tm1))
        print('h_t', h_t.size())
        print('c_t', c_t.size())
        # compute sentinel vector
        # could either concat h_tm1 with x or have to gates
        sv = F.sigmoid(self.h_gate(h_tm1) + self.x_gate(x))
        s_t = sv * F.tanh(c_t)
        print('s_t', s_t.size())
        return h_t, c_t, s_t


class ImageEncoder(nn.Module):

    def __init__(self, input_shape, hidden_size, embedding_size):
        super(ImageEncoder, self).__init__()
        print('input shape', input_shape)
        print('hidden_size', hidden_size)
        self.average_pool = nn.AvgPool2d(input_shape[0])
        # affine transformation of attention features
        self.v_affine = nn.Linear(input_shape[2], hidden_size)
        # affine transformation of global image features
        self.global_affine = nn.Linear(input_shape[2], embedding_size)

    def forward(self, x):
        # x = V, (batch_size, 8, 8, 1536)
        print('Image Encoder')
        input_shape = x.size()
        pixels = input_shape[1] * input_shape[2]  # 8x8 = 64
        global_image = self.average_pool(x).view(input_shape[0], -1)
        print('global a', global_image.size())
        inputs = x.view(input_shape[0], pixels, input_shape[3])

        # transform
        global_image = self.global_affine(global_image)
        print('global v', global_image.size())
        inputs = self.v_affine(inputs)
        print('image regions', inputs.size())

        return global_image, inputs


class MultimodalDecoder(nn.Module):

    def __init__(self, input_shape, hidden_size, n=0):
        super(MultimodalDecoder, self).__init__()
        self.layers = []
        if n:
            new_input_shape = input_shape*2
            self.layers.append(nn.Linear(input_shape, input_shape))
            input_shape = new_input_shape
        else:
            n = 1

        for _ in range(n - 1):
            self.layers.append(nn.Linear(input_shape, input_shape))

        # output layer, this is the only layer if n=0
        self.output_layer = nn.Linear(input_shape, hidden_size)

    def forward(self, x):
        # x: context_vector + h_t (batch_size, hidden_size)
        print('Multimodal Decoder')
        concat = x
        for layer in self.layers:
            y = F.relu(layer(concat))
            concat = torch.cat((concat, y), dim=1)

        # softmax on output
        y = F.softmax(self.output_layer(concat))
        return y


class AttentionLayer(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(AttentionLayer, self).__init__()
        # input_size 512
        # hidden_size 512

        self.v_att = nn.Linear(input_size, hidden_size)

        self.s_proj = nn.Linear(input_size, input_size)
        self.s_att = nn.Linear(input_size, hidden_size)

        self.h_proj = nn.Linear(input_size, input_size)
        self.h_att = nn.Linear(input_size, hidden_size)

        self.alpha_layer = nn.Linear(hidden_size, 1)
        # might move this outside
        self.context_proj = nn.Linear(input_size, input_size)

    def forward(self, x):
        # x : [V, s_t, h_t]
        # V = [v1, v2, ..., vk]
        # c_t is context vector: sum of alphas*v
        # output should be beta*s_t + (1-beta)*c_t
        print('attention layer')
        V = x[0]  # (batch_size, 8x8, hidden_size)
        s_t = x[1]  # (batch_size, hidden_size)
        h_t = x[2]  # (batch_size, hidden_size)

        # embed visual features
        v_embed = F.relu(self.v_att(V))  # (batch_size, 64, hidden_size)

        # s_t embedding
        s_proj = F.relu(self.s_proj(s_t))  # (batch_size, hidden_size)
        s_att = self.s_att(s_proj)  # (batch_size, hidden_size)

        # h_t embedding
        h_proj = F.tanh(self.h_proj(h_t))  # (batch_size, hidden_size)
        h_att = self.h_att(h_proj)  # (batch_size, hidden_size)

        # make s_proj the same dimension as V
        s_proj = s_proj.unsqueeze(1)  # (batch_size, 1, hidden_size)

        # make s_att the same dimension as v_att
        s_att = s_att.unsqueeze(1)  # (batch_size, 1, hidden_size)

        # make h_att the same dimension as regions_att
        h_att = h_att.unsqueeze(1).expand(h_att.size()[0],
                                          V.size()[1] + 1,
                                          h_att.size()[1])
        # (batch_size, 64 + 1, hidden_size)

        # concatenations
        regions = torch.cat((V, s_proj), dim=1)
        # (batch_size, 64 +1, hidden_size)
        regions_att = torch.cat((v_embed, s_att), dim=1)
        # (batch_size, 64 +1, hidden_size)

        # add h_t to regions_att
        alpha_input = F.tanh(regions_att + h_att)
        # (batch_size, 64 +1, hidden_size)

        # compute alphas + beta
        alpha = F.softmax(self.alpha_layer(alpha_input).squeeze(2), dim=1)
        # (batch_size, 64 + 1)
        alpha = alpha.unsqueeze(2)  # (batch_size, 64 +1, 1)

        # multiply with regions
        context_vector = (alpha * regions).sum(dim=1)  # the actual z_t
        # (batch_size, hidden_size)

        z_t = F.tanh(self.context_proj(context_vector + h_proj))
        # (batch_size, hidden_size)

        return z_t
