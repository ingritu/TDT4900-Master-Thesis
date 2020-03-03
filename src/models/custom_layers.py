import torch
import torch.nn as nn
import torch.nn.functional as F


class SentinelLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, n=0):
        """
        Implementation of the Sentinel LSTM by Lu et al. Knowing when to look.
        Also has the option to stack LSTMCells before generating the
        Sentinel vector. There are skip connections between all LSTMCells in
        the stack.

        Parameters
        ----------
        input_size : int.
        hidden_size : int.
        n : int. The number of extra LSTM cells to use. Default is 0.
        """
        super(SentinelLSTM, self).__init__()
        # NB! there is a difference between LSTMCell and LSTM.
        # LSTM is notably much quicker
        self.n = n
        self.lstm_cells = []
        if n > 0:
            inp_size = hidden_size
        else:
            inp_size = input_size
        self.lstm_kernel = nn.LSTMCell(inp_size, hidden_size)
        self.x_gate = nn.Linear(inp_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size, hidden_size)

        inp_size = input_size
        hid_size = hidden_size
        for _ in range(self.n):
            self.lstm_cells.append(nn.LSTMCell(inp_size, hid_size))
            inp_size = hid_size

    def forward(self, x, states):
        # print('Sentinel LSTM')
        # remember old states
        h_tm1, c_tm1 = states
        # new states lists
        hs = torch.zeros(h_tm1.size())
        cs = torch.zeros(c_tm1.size())
        inputs = x
        for i in range(self.n):
            # feed layers the correct h and c states
            h, c = self.lstm_cells[i](inputs, (h_tm1[i], c_tm1[i]))
            hs[i] = h
            cs[i] = c
            # add residual
            inputs = h + inputs

        # get new states
        h_t, c_t = self.lstm_kernel(inputs, (h_tm1[-1], c_tm1[-1]))
        hs[-1] = h_t
        cs[-1] = c_t

        # compute sentinel vector
        # could either concat h_tm1 with x or have to gates
        sv = torch.sigmoid(self.h_gate(h_tm1[-1]) + self.x_gate(inputs))
        s_t = sv * torch.tanh(c_t)
        return hs, cs, h_t, s_t


class ImageEncoder(nn.Module):

    def __init__(self, input_shape, hidden_size, embedding_size):
        """
        Reshapes or embeds the image features more to fit dims of
        the rest of the model.

        Parameters
        ----------
        input_shape : int.
        hidden_size : int.
        embedding_size : int.
        """
        super(ImageEncoder, self).__init__()
        self.average_pool = nn.AvgPool2d(input_shape[0])
        # affine transformation of attention features
        self.v_affine = nn.Linear(input_shape[2], hidden_size)
        # affine transformation of global image features
        self.global_affine = nn.Linear(input_shape[2], embedding_size)

    def forward(self, x):
        # x = V, (batch_size, 8, 8, 1536)
        # print('Image Encoder')
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
        """
        Multimodal decoding part of the model.

        Parameters
        ----------
        input_shape : int.
        hidden_size : int.
        n : int. Default is 0.
        """
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
        # print('Multimodal Decoder')
        concat = x
        for layer in self.layers:
            y = F.relu(layer(concat))
            concat = torch.cat((concat, y), dim=1)

        # softmax on output
        y = torch.softmax(self.output_layer(concat), dim=1)
        return y


class AttentionLayer(nn.Module):

    def __init__(self, input_size, hidden_size):
        """
        Implementation of the Soft visual attention block
        by Lu et al. Knowing when to look.

        Parameters
        ----------
        input_size : int.
        hidden_size : int.
        """
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
        # print('attention layer')
        V = x[0]  # (batch_size, 8x8, hidden_size)
        s_t = x[1]  # (batch_size, hidden_size)
        h_t = x[2]  # (batch_size, hidden_size)

        # embed visual features
        v_embed = F.relu(self.v_att(V))  # (batch_size, 64, hidden_size)

        # s_t embedding
        s_proj = F.relu(self.s_proj(s_t))  # (batch_size, hidden_size)
        s_att = self.s_att(s_proj)  # (batch_size, hidden_size)

        # h_t embedding
        h_proj = torch.tanh(self.h_proj(h_t))  # (batch_size, hidden_size)
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
        alpha_input = torch.tanh(regions_att + h_att)
        # (batch_size, 64 +1, hidden_size)

        # compute alphas + beta
        alpha = torch.softmax(self.alpha_layer(alpha_input).squeeze(2), dim=1)
        # (batch_size, 64 + 1)
        alpha = alpha.unsqueeze(2)  # (batch_size, 64 +1, 1)

        # multiply with regions
        context_vector = (alpha * regions).sum(dim=1)  # the actual z_t
        # (batch_size, hidden_size)

        z_t = torch.tanh(self.context_proj(context_vector + h_proj))
        # (batch_size, hidden_size)

        return z_t
