import torch
from torch import nn as nn
from torch.nn import functional as F

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
                 embedding_size=300,
                 seed=222):
        super(AdaptiveModel, self).__init__()
        self.input_shape = input_shape
        self.visual_feature_shape = input_shape[0]
        self.max_len = input_shape[1]
        self.hidden_size = hidden_size

        self.vocab_size = vocab_size
        self.em_size = embedding_size
        self.random_seed = seed

        # layers
        self.image_encoder = ImageEncoder(self.visual_feature_shape,
                                          self.hidden_size,
                                          self.em_size)
        self.embedding = nn.Embedding(self.vocab_size, self.em_size)
        self.sentinel_lstm = SentinelLSTM(self.em_size * 2, self.hidden_size)
        self.attention_block = AttentionLayer(self.hidden_size,
                                              self.hidden_size)
        self.decoder = MultimodalDecoder(self.hidden_size,
                                         self.vocab_size, n=1)

    def initialize_variables(self, batch_size):
        # initialize h and c as zeros
        h_0 = torch.zeros(batch_size, self.hidden_size)
        c_0 = torch.zeros(batch_size, self.hidden_size)
        return h_0, c_0

    def forward(self, x, caption_lengths):
        # visual features (batch_size, 8 ,8, 1536)
        # batch_size is equal to the number of images
        im_input = x[0]
        w_input = x[1]

        global_images, encoded_images = self.image_encoder(im_input)
        # (batch_size, embedding_size) (batch, 512) global_images
        # (batch_size, region_size, hidden_size) (batch, 64, 512) encoded_imgs

        embedded_w = self.embedding(w_input)  # (batch, max_len, hidden_size)

        batch_size = encoded_images.size()[0]

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
        decoding_lengths = (caption_lengths - 1)

        for timestep in range(self.max_len):

            x_t = inputs[:batch_size, timestep, :]

            h_t, c_t, s_t = self.sentinel_lstm(x_t, (h_t[:batch_size],
                                                     c_t[:batch_size]))

            z_t = self.attention_block([encoded_images, s_t, h_t])

            pt = self.decoder([z_t, h_t])
            predictions[:batch_size, timestep, :] = pt

        return predictions


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

