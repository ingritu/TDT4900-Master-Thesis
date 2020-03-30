import torch
import torch.nn as nn
import torch.nn.functional as f


class SentinelLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        """
        Implementation of the Sentinel LSTM by Lu et al. Knowing when to look.

        Parameters
        ----------
        input_size : int.
        hidden_size : int.
        """
        super(SentinelLSTM, self).__init__()
        # NB! there is a difference between LSTMCell and LSTM.
        # LSTM is notably much quicker
        self.lstm_cells = []
        self.lstm_kernel = nn.LSTMCell(input_size, hidden_size)
        self.x_gate = nn.Linear(input_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, states):
        # print('Sentinel LSTM')
        # remember old states
        h_tm1, c_tm1 = states

        # get new states
        h_t, c_t = self.lstm_kernel(x, (h_tm1, c_tm1))

        # compute sentinel vector
        # could either concat h_tm1 with x or have to gates
        sv = torch.sigmoid(self.h_gate(h_tm1) + self.x_gate(x))
        s_t = sv * torch.tanh(c_t)
        return h_t, c_t, s_t


class ImageEncoder(nn.Module):

    def __init__(self, input_shape, hidden_size, embedding_size, dr=0.5):
        """
        Reshapes or embeds the image features more to fit dims of
        the rest of the model.

        Parameters
        ----------
        input_shape : int.
        hidden_size : int.
        embedding_size : int.
        dr : float. Dropout value.
        """
        super(ImageEncoder, self).__init__()
        self.average_pool = nn.AvgPool2d(input_shape[0])
        # affine transformation of attention features
        self.v_affine = nn.Linear(input_shape[2], hidden_size)
        self.v_dr = nn.Dropout(dr)
        # affine transformation of global image features
        self.global_affine = nn.Linear(input_shape[2], embedding_size)
        self.g_dr = nn.Dropout(dr)

    def forward(self, x):
        # x = V, (batch_size, 8, 8, 1536)
        # print('Image Encoder')
        input_shape = x.size()
        pixels = input_shape[1] * input_shape[2]  # 8x8 = 64
        global_image = self.average_pool(x).view(input_shape[0], -1)
        inputs = x.view(input_shape[0], pixels, input_shape[3])

        # transform
        global_image = f.relu(self.global_affine(global_image))
        global_image = self.g_dr(global_image)
        inputs = f.relu(self.v_affine(inputs))
        inputs = self.v_dr(inputs)

        return global_image, inputs


class MultimodalDecoder(nn.Module):

    def __init__(self, input_shape, hidden_size):
        """
        Multimodal decoding part of the model.

        Parameters
        ----------
        input_shape : int.
        hidden_size : int.
        """
        super(MultimodalDecoder, self).__init__()
        # output layer, this is the only layer if n=0
        self.output_layer = nn.Linear(input_shape, hidden_size)

    def forward(self, x):
        # x: context_vector + h_t (batch_size, hidden_size)
        # print('Multimodal Decoder')
        y = self.output_layer(x)
        return y


class AttentionLayer(nn.Module):

    def __init__(self, input_size, hidden_size, dr=0.5):
        """
        Implementation of the Soft visual attention block
        by Lu et al. Knowing when to look.

        Parameters
        ----------
        input_size : int.
        hidden_size : int.
        dr : float. Dropout value.
        """
        super(AttentionLayer, self).__init__()
        # input_size 512
        # hidden_size 512

        self.v_att = nn.Linear(input_size, hidden_size)

        self.s_proj = nn.Linear(input_size, input_size)
        self.s_proj_dr = nn.Dropout(dr)
        self.s_att = nn.Linear(input_size, hidden_size)

        self.h_proj = nn.Linear(input_size, input_size)
        self.h_proj_dr = nn.Dropout(dr)
        self.h_att = nn.Linear(input_size, hidden_size)

        self.alpha_layer = nn.Linear(hidden_size, 1)
        self.alpha_dr = nn.Dropout(dr)
        # might move this outside
        self.context_proj = nn.Linear(input_size, input_size)
        self.context_proj_dr = nn.Dropout(dr)

    def forward(self, x):
        # x : [V, s_t, h_t]
        # V = [v1, v2, ..., vk]
        # c_t is context vector: sum of alphas*v
        # output should be beta*s_t + (1-beta)*c_t
        # print('attention layer')
        v = x[0]  # (batch_size, 8x8, hidden_size)
        s_t = x[1]  # (batch_size, hidden_size)
        h_t = x[2]  # (batch_size, hidden_size)

        # embed visual features
        v_embed = f.relu(self.v_att(v))  # (batch_size, 64, hidden_size)

        # s_t embedding
        s_proj = f.relu(self.s_proj(s_t))  # (batch_size, hidden_size)
        s_proj = self.s_proj_dr(s_proj)  # dropout
        s_att = self.s_att(s_proj)  # (batch_size, hidden_size)

        # h_t embedding
        h_proj = torch.tanh(self.h_proj(h_t))  # (batch_size, hidden_size)
        h_proj = self.h_proj_dr(h_proj)  # dropout
        h_att = self.h_att(h_proj)  # (batch_size, hidden_size)

        # make s_proj the same dimension as V
        s_proj = s_proj.unsqueeze(1)  # (batch_size, 1, hidden_size)

        # make s_att the same dimension as v_att
        s_att = s_att.unsqueeze(1)  # (batch_size, 1, hidden_size)

        # make h_att the same dimension as regions_att
        h_att = h_att.unsqueeze(1).expand(h_att.size()[0],
                                          v.size()[1] + 1,
                                          h_att.size()[1])
        # (batch_size, 64 + 1, hidden_size)

        # concatenations
        regions = torch.cat((v, s_proj), dim=1)
        # (batch_size, 64 +1, hidden_size)
        regions_att = torch.cat((v_embed, s_att), dim=1)
        # (batch_size, 64 +1, hidden_size)

        # add h_t to regions_att
        alpha_input = torch.tanh(regions_att + h_att)
        # (batch_size, 64 +1, hidden_size)
        alpha_input = self.alpha_dr(alpha_input)  # dropout

        # compute alphas + beta
        alpha = self.alpha_layer(alpha_input).squeeze(2)
        # (batch_size, 64 + 1)
        alpha = torch.softmax(alpha, dim=1)
        # (batch_size, 64 + 1)
        alpha = alpha.unsqueeze(2)  # (batch_size, 64 +1, 1)

        # multiply with regions
        context_vector = (alpha * regions).sum(dim=1)  # the actual z_t
        # (batch_size, hidden_size)

        z_t = torch.tanh(self.context_proj(context_vector + h_proj))
        z_t = self.context_proj_dr(z_t)
        # (batch_size, hidden_size)

        return z_t
