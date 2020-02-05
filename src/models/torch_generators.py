import torch
from torch import nn as nn
from torch.nn import functional as F


def model_switcher(model_str):
    model_str = model_str.lower()
    switcher = {
        'Tutorial': TutorialModel,
        'adaptive': AdaptiveModel,
        'default': TutorialModel
    }
    return switcher[model_str]


class TutorialModel(nn.Module):

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


class AdaptiveModel(nn.Module):

    def __init__(self):
        # TODO: implement this
        super(AdaptiveModel, self).__init__()

    def forward(self, x):
        # TODO: implement this
        pass


def flatten_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

