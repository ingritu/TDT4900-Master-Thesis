import torch
import torch.nn as nn
import torch.nn.functional as f


class SentinelLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dr=0.5):
        """
        Implementation of the Sentinel LSTM by Lu et al. Knowing when to look.

        Parameters
        ----------
        input_size : int.
        hidden_size : int.
        dr : float. Dropout value.
        """
        super(SentinelLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dr = dr
        self.lstm_kernel = nn.LSTMCell(self.input_size, self.hidden_size)
        self.x_gate = nn.Linear(self.input_size, self.hidden_size)
        self.h_gate = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x, states):
        # remember old states
        h_tm1, c_tm1 = states
        # h_tm1, c_tm1 = h_tm1.squeeze(1), c_tm1.squeeze(1)

        # get new states
        h_t, c_t = self.lstm_kernel(x, (h_tm1, c_tm1))

        # compute sentinel vector
        sv = torch.sigmoid(self.h_gate(h_tm1) + self.x_gate(x))
        s_t = sv * torch.tanh(c_t)
        return h_t, c_t, h_t, s_t


class SentinelLSTM2(nn.Module):

    def __init__(self, input_size, hidden_size, dr=0.5):
        """
        Contains 2 LSTMCells

        Parameters
        ----------
        input_size
        hidden_size
        dr
        """
        super(SentinelLSTM2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dr = dr

        self.sentinel_lstm = SentinelLSTM(self.input_size,
                                          self.hidden_size,
                                          dr=self.dr)
        self.lstm_cell_0 = nn.LSTMCell(self.input_size, self.hidden_size)

    def forward(self, x, states):
        # unpack states
        h_tm1, c_tm1 = states

        old_states_0 = (h_tm1[:, 0], c_tm1[:, 0])
        h0, c0 = self.lstm_cell_0(x, old_states_0)

        # residual connection
        x = h0.repeat(1, 2) + x

        # get new states
        old_states_1 = (h_tm1[:, 1], c_tm1[:, 1])

        h_t, c_t, h_top, s_t = self.sentinel_lstm(x, old_states_1)

        ht = torch.stack((h0, h_top), dim=1)
        ct = torch.stack((c0, c_t.squeeze(1)), dim=1)
        return ht, ct, h_top, s_t


class SentinelLSTM3(nn.Module):

    def __init__(self, input_size, hidden_size, dr=0.5):
        """
        Contains 3 LSTMCells
        Parameters
        ----------
        input_size
        hidden_size
        dr
        """
        super(SentinelLSTM3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dr = dr

        self.lstm_cell_0 = nn.LSTMCell(self.input_size, self.hidden_size)

        self.lstm_cell_1 = nn.LSTMCell(self.input_size, self.hidden_size)

        self.sentinel_lstm = SentinelLSTM(self.input_size,
                                          self.hidden_size,
                                          dr=self.dr)

    def forward(self, x, states):
        # unpack states
        h_tm1, c_tm1 = states

        old_states_0 = (h_tm1[:, 0], c_tm1[:, 0])
        h0, c0 = self.lstm_cell_0(x, old_states_0)

        # residual connection
        x = h0.repeat(1, 2) + x

        old_states_1 = (h_tm1[:, 1], c_tm1[:, 1])
        h1, c1 = self.lstm_cell_1(x, old_states_1)

        # residual connection
        x = h1.repeat(1, 2) + x

        # get new states
        old_states_2 = (h_tm1[:, 2], c_tm1[:, 2])
        h_t, c_t, h_top, s_t = self.sentinel_lstm(x, old_states_2)

        ht = torch.stack((h0, h1, h_top), dim=1)
        ct = torch.stack((c0, c1, c_t.squeeze(1)), dim=1)
        return ht, ct, h_top, s_t


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


class BasicImageEncoder(nn.Module):

    def __init__(self, input_shape, embedding_size, dr=0.5):
        """

        Parameters
        ----------
        input_shape
        embedding_size
        dr
        """
        super(BasicImageEncoder, self).__init__()
        self.average_pool = nn.AvgPool2d(input_shape[0])
        # affine transformation of global image features
        self.global_affine = nn.Linear(input_shape[2], embedding_size)
        self.g_dr = nn.Dropout(dr)

    def forward(self, x):
        input_shape = x.size()
        global_image = self.average_pool(x).view(input_shape[0], -1)

        # transform
        global_image = f.relu(self.global_affine(global_image))
        global_image = self.g_dr(global_image)
        return global_image, torch.empty(global_image.size())


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
