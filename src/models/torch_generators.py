import torch
from torch import nn as nn
import numpy as np

from src.models.custom_layers import SentinelLSTM
from src.models.custom_layers import AttentionLayer
from src.models.custom_layers import ImageEncoder
from src.models.custom_layers import MultimodalDecoder


def model_switcher(model_str):
    model_str = model_str.lower()
    switcher = {
        'adaptive': AdaptiveModel,
        'adaptive_decoder': [AdaptiveDecoder, ImageEncoder]
    }
    return switcher.get(model_str, AdaptiveModel)


class AdaptiveModel(nn.Module):

    def __init__(self,
                 input_shape,
                 hidden_size,
                 vocab_size,
                 device,
                 num_lstms=0,
                 embedding_size=300,
                 seed=222):
        super(AdaptiveModel, self).__init__()
        self.input_shape = input_shape
        self.visual_feature_shape = input_shape[0]
        self.max_len = input_shape[1]
        self.hidden_size = hidden_size

        self.vocab_size = vocab_size
        self.em_size = embedding_size
        self.num_lstms = num_lstms
        self.random_seed = seed

        self.device = device

        # layers
        # encoder
        self.image_encoder = ImageEncoder(self.visual_feature_shape,
                                          self.hidden_size,
                                          self.em_size)
        # decoder
        self.decoder = AdaptiveDecoder(self.input_shape,
                                       self.hidden_size,
                                       self.vocab_size,
                                       self.device,
                                       num_lstms=self.num_lstms,
                                       embedding_size=self.em_size,
                                       seed=self.random_seed)

    def initialize_variables(self, batch_size):
        # initialize h and c as zeros
        hs = torch.zeros(self.num_lstms + 1, batch_size, self.hidden_size)\
            .to(self.device)
        cs = torch.zeros(self.num_lstms + 1, batch_size, self.hidden_size)\
            .to(self.device)
        return hs, cs

    def forward(self, x, caption_lengths, has_end_seq_token=True):
        # visual features (batch_size, 8 ,8, 1536)
        # batch_size is equal to the number of captions
        im_input = x[0].to(self.device)
        w_input = x[1].to(self.device)

        global_images, encoded_images = self.image_encoder(im_input)
        # (batch_size, embedding_size) (batch, 512) global_images
        # (batch_size, region_size, hidden_size) (batch, 64, 512) encoded_imgs

        # sort batches by caption length descending, this way the whole
        # batch_size_t will be correct
        # convert to tensor
        caption_lengths = torch.from_numpy(caption_lengths)
        caption_lengths, sort_idx = caption_lengths.sort(dim=0,
                                                         descending=True)
        w_input = w_input[sort_idx]  # (batch_size, max_len)
        global_images = global_images[sort_idx]  # (batch_size, embedding_size)
        encoded_images = encoded_images[sort_idx]  # (batch_size, 64, 1536)

        batch_size = encoded_images.size()[0]

        decoding_lengths = np.copy(caption_lengths)
        if has_end_seq_token:
            decoding_lengths = (decoding_lengths - 1)
        batch_max_length = max(decoding_lengths)

        predictions = torch.zeros(batch_size,
                                  self.max_len,
                                  self.vocab_size)
        # initialize h and c
        h_t, c_t = self.decoder.initialize_variables(batch_size)

        for timestep in range(batch_max_length):
            batch_size_t = sum([lens > timestep for lens in decoding_lengths])
            # x: [input_img, input_w]
            # input_img: [global_image, encoded_image]
            # image features does not vary over time
            input_image_t = [global_images[:batch_size_t],
                             encoded_images[:batch_size_t]]
            x_t = [input_image_t, w_input[:batch_size_t, timestep]]

            pt, h_t, c_t = self.decoder(x_t,
                                        (h_t[:, :batch_size_t],
                                         c_t[:, :batch_size_t]))
            predictions[:batch_size_t, timestep, :] = pt

        return predictions, decoding_lengths


class AdaptiveDecoder(nn.Module):

    """
    Adaptive Decoder.

    This class will not do any encoding of the images, but expects an
    encoded image as input. This generator does not output full sequences,
    instead it only outputs the predictions at a timestep.
    """

    def __init__(self,
                 input_shape,
                 hidden_size,
                 vocab_size,
                 device,
                 num_lstms=0,
                 embedding_size=300,
                 seed=222):
        super(AdaptiveDecoder, self).__init__()

        self.input_shape = input_shape
        self.visual_feature_shape = input_shape[0]
        self.max_len = input_shape[1]
        self.hidden_size = hidden_size

        self.vocab_size = vocab_size
        self.em_size = embedding_size
        self.num_lstms = num_lstms
        self.random_seed = seed

        self.device = device

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.em_size)
        self.sentinel_lstm = SentinelLSTM(self.em_size * 2,
                                          self.hidden_size,
                                          n=self.num_lstms)
        self.attention_block = AttentionLayer(self.hidden_size,
                                              self.hidden_size)
        self.decoder = MultimodalDecoder(self.hidden_size,
                                         self.vocab_size, n=1)

    def forward(self, x, states):
        # unpack input
        input_img, input_w = x
        print('input_w', input_w.size())
        global_image, encoded_images = input_img
        # global (batch_size, embedding_size)
        # encoded (batch_size, 64, hidden_size)

        # embed word
        embedded_w = self.embedding(input_w)
        # (batch_size, embedding_size)

        # cat input w with v_avg
        print('w_em', embedded_w.size())
        print('g_img', global_image.size())
        x_t = torch.cat((embedded_w, global_image), dim=1)
        # (batch_size, embedding_size*2)

        # get states
        h_tm1, c_tm1 = states

        # decoding
        h_t, c_t, h_top, s_t = self.sentinel_lstm(x_t, (h_tm1, c_tm1))
        z_t = self.attention_block([encoded_images, s_t, h_top])
        pt = self.decoder(z_t)

        return pt, h_t, c_t

    def initialize_variables(self, batch_size):
        # initialize h and c as zeros
        hs = torch.zeros(self.num_lstms + 1, batch_size, self.hidden_size) \
            .to(self.device)
        cs = torch.zeros(self.num_lstms + 1, batch_size, self.hidden_size) \
            .to(self.device)
        return hs, cs



