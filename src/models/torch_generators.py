import torch
from torch import nn as nn
from torch.nn import functional as F


class TutorialModel(nn.module):

    def __init__(self,
                 input_shape,
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
        self.w_em1 = nn.Embedding(self.max_len, self.embedding_size)

        self.lstm = nn.LSTM(self.embedding_size, 256)

        self.fc1 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, self.vocab_size)

    def forward(self, x):
        # put everything together here
        # prepare inputs
        im_input = x[0]
        w_input = x[1]

        im_vector = F.relu(self.im_fc1(im_input))

        embbedding_vector = self.w_em1(w_input)
        lstm_vector = self.lstm(embbedding_vector)

        sum_vector = im_vector + lstm_vector

        fc_output = F.relu(self.fc1(sum_vector))
        output = F.softmax(self.output_layer(fc_output))

        # does the next x input have to be the output here???
        return output



