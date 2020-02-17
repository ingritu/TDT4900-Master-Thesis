import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from src.models.custom_layers import SentinelLSTM
from src.models.custom_layers import AttentionLayer
from src.models.custom_layers import ImageEncoder
from src.models.custom_layers import MultimodalDecoder


def model_switcher(model_str):
    model_str = model_str.lower()
    switcher = {
        'Tutorial': TutorialModel,
        'adaptive': AdaptiveModel,
        'default': TutorialModel
    }
    return switcher[model_str]


class AdaptiveModel(nn.Module):

    def __init__(self,
                 input_shape,
                 hidden_size,
                 vocab_size,
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

        # layers
        self.image_encoder = ImageEncoder(self.visual_feature_shape,
                                          self.hidden_size,
                                          self.em_size)
        self.embedding = nn.Embedding(self.vocab_size, self.em_size)
        self.sentinel_lstm = SentinelLSTM(self.em_size * 2,
                                          self.hidden_size,
                                          n=self.num_lstms)
        self.attention_block = AttentionLayer(self.hidden_size,
                                              self.hidden_size)
        self.decoder = MultimodalDecoder(self.hidden_size,
                                         self.vocab_size, n=1)

    def initialize_variables(self, batch_size):
        # initialize h and c as zeros
        hs = torch.zeros(self.num_lstms + 1, batch_size, self.hidden_size)
        cs = torch.zeros(self.num_lstms + 1, batch_size, self.hidden_size)
        return hs, cs

    def forward(self, x, caption_lengths, has_end_seq_token=True):
        # visual features (batch_size, 8 ,8, 1536)
        # batch_size is equal to the number of images
        im_input = x[0]
        w_input = x[1]

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

        embedded_w = self.embedding(w_input)  # (batch, max_len, hidden_size)

        batch_size = encoded_images.size()[0]

        decoding_lengths = np.copy(caption_lengths)
        if has_end_seq_token:
            decoding_lengths = (decoding_lengths - 1)
        batch_max_length = max(decoding_lengths)

        # replicate global image
        global_images = global_images.unsqueeze(1).expand_as(embedded_w)
        # concat w_t with v_avg
        inputs = torch.cat((embedded_w, global_images), dim=2)
        # (batch_size, max_len, embedding_size * 2)

        predictions = torch.zeros(batch_size,
                                  self.max_len,
                                  self.vocab_size)
        # initialize h and c
        h_t, c_t = self.initialize_variables(batch_size)

        for timestep in range(batch_max_length):
            batch_size_t = sum([l > timestep for l in decoding_lengths])
            x_t = inputs[:batch_size_t, timestep, :]

            h_t, c_t, h_top, s_t = self.sentinel_lstm(
                                                    x_t,
                                                    (h_t[:, :batch_size_t, :],
                                                     c_t[:, :batch_size_t, :]))

            z_t = self.attention_block([encoded_images[:batch_size_t],
                                        s_t,
                                        h_top])

            pt = self.decoder(z_t)
            predictions[:batch_size_t, timestep, :] = pt

        return predictions, decoding_lengths


class TutorialModel(nn.Module):

    def __init__(self,
                 input_shape,
                 hidden,
                 vocab_size,
                 embedding_size=300,
                 seed=222):
        super(TutorialModel, self).__init__()
        self.input_shape = input_shape
        self.img_feature_shape = self.input_shape[0]
        self.max_len = self.input_shape[1]

        self.vocab_size = vocab_size

        self.embedding_size = embedding_size
        self.random_seed = seed

        self.model_name = 'TutorialModel'

        # add layers here
        self.im_fc1 = nn.Linear(self.img_feature_shape, 256)
        self.w_em1 = nn.Embedding(self.vocab_size, self.embedding_size)

        self.lstm = nn.LSTM(self.embedding_size, 256)
        self.w_fc1 = nn.Linear(9472, 256)
        self.fc1 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, self.vocab_size)

    def forward(self, x):
        # put everything together here
        # prepare inputs
        im_input = x[0]
        # print(im_input.size())
        w_input = x[1]
        # print(w_input.size())

        global_image = F.relu(self.im_fc1(im_input))

        embbedding_vector = self.w_em1(w_input)
        # print('embedding', embbedding_vector.size())
        lstm_vector, _ = self.lstm(embbedding_vector)

        # print('image', global_image.size())
        # print('lstm', lstm_vector.size())

        lstm_vector = lstm_vector.view(-1, flatten_features(lstm_vector))
        lstm_vector = F.relu(self.w_fc1(lstm_vector))
        # print(lstm_vector.size())
        sum_vector = lstm_vector + global_image

        fc_output = F.relu(self.fc1(sum_vector))
        output = F.softmax(self.output_layer(fc_output), dim=1)

        return output


def flatten_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

