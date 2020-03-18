import torch
from torch import nn as nn

from src.models.custom_layers import SentinelLSTM
from src.models.custom_layers import AttentionLayer
from src.models.custom_layers import ImageEncoder
from src.models.custom_layers import MultimodalDecoder


def model_switcher(model_str):
    model_str = model_str.lower()
    switcher = {
        'adaptive': AdaptiveModel,
        'basic': BasicModel,
    }
    return switcher.get(model_str, AdaptiveModel)


class AbstractModel(nn.Module):

    def __init__(self,
                 input_shape,
                 max_len,
                 hidden_size,
                 vocab_size,
                 device,
                 embedding_size=512):
        """
        Generic Abstract Model

        Parameters
        ----------
        input_shape : list
        max_len : int.
        hidden_size : int.
        vocab_size : int.
        device : torch.device.
        embedding_size : int.
        """
        super(AbstractModel, self).__init__()
        self.visual_feature_shape = input_shape
        self.max_len = max_len
        self.hidden_size = hidden_size

        self.vocab_size = vocab_size
        self.em_size = embedding_size

        self.device = device

        # layers
        # encoder
        self.encoder = ImageEncoder(self.visual_feature_shape,
                                    self.hidden_size,
                                    self.em_size)
        # decoder
        self.decoder = None

    def forward(self, x, caption_lengths, has_end_seq_token=True):
        # visual features (batch_size, 8, 8, 1536)
        # batch_size is equal to the number of captions
        im_input = x[0].to(self.device)
        w_input = x[1].to(self.device)

        global_images, encoded_images = self.encoder(im_input)
        # (batch_size, embedding_size) (batch, 512) global_images
        # (batch_size, region_size, hidden_size) (batch, 64, 512) encoded_imgs

        # sort batches by caption length descending, this way the whole
        # batch_size_t will be correct
        # convert to tensor
        caption_lengths = torch.from_numpy(caption_lengths).to(self.device)
        caption_lengths, sort_idx = caption_lengths.sort(dim=0,
                                                         descending=True)
        w_input = w_input[sort_idx]  # (batch_size, max_len)
        global_images = global_images[sort_idx]  # (batch_size, embedding_size)
        encoded_images = encoded_images[sort_idx]  # (batch_size, 64, 1536)

        target = w_input[:, 1:]  # sorted targets
        target = target.to(self.device)

        batch_size = encoded_images.size()[0]

        decoding_lengths = caption_lengths
        if has_end_seq_token:
            decoding_lengths -= 1
        batch_max_length = int(torch.max(decoding_lengths))

        predictions = torch.zeros(batch_size,
                                  self.max_len,
                                  self.vocab_size).to(self.device)
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
                                        (h_t[:batch_size_t],
                                         c_t[:batch_size_t]))
            predictions[:batch_size_t, timestep, :] = torch.softmax(pt, dim=1)

        return predictions, target, decoding_lengths


class AdaptiveModel(AbstractModel):

    def __init__(self,
                 input_shape,
                 max_len,
                 hidden_size,
                 vocab_size,
                 device,
                 embedding_size=512):
        """
        Adaptive Model

        Parameters
        ----------
        input_shape : list
        max_len : int.
        hidden_size : int.
        vocab_size : int.
        device : torch.device.
        embedding_size : int.
        """
        super(AdaptiveModel, self).__init__(input_shape, max_len, hidden_size,
                                            vocab_size, device,
                                            embedding_size=embedding_size)
        # decoder
        self.decoder = AdaptiveDecoder(
            self.max_len,
            self.hidden_size,
            self.em_size,
            self.vocab_size,
            self.device)


class AdaptiveDecoder(nn.Module):

    def __init__(self,
                 max_len,
                 hidden_size,
                 embedding_size,
                 vocab_size,
                 device):
        """
        Adaptive Decoder.

        This class will not do any encoding of the images, but expects an
        encoded image as input. This generator does not output full sequences,
        instead it only outputs the predictions at a timestep.

        Parameters
        ----------
        max_len : int.
        hidden_size : int.
        embedding_size : int.
        vocab_size : int.
        device : torch.device.
        """
        super(AdaptiveDecoder, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.em_size = embedding_size

        # device
        self.device = device

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.em_size)
        self.sentinel_lstm = SentinelLSTM(self.em_size * 2,
                                          self.hidden_size)
        self.attention_block = AttentionLayer(self.hidden_size,
                                              self.hidden_size)
        self.decoder = MultimodalDecoder(self.hidden_size,
                                         self.vocab_size)

    def forward(self, x, states):
        # unpack input
        input_img, input_w = x
        global_image, encoded_images = input_img
        # global (batch_size, embedding_size)
        # encoded (batch_size, 64, hidden_size)

        # embed word
        embedded_w = self.embedding(input_w)
        # (batch_size, embedding_size)

        # cat input w with v_avg
        x_t = torch.cat((embedded_w, global_image), dim=1)
        # (batch_size, embedding_size*2)

        # get states
        h_tm1, c_tm1 = states

        # decoding
        h_t, c_t, s_t = self.sentinel_lstm(x_t, (h_tm1, c_tm1))
        z_t = self.attention_block([encoded_images, s_t, h_t])
        pt = self.decoder(z_t)

        return pt, h_t, c_t

    def initialize_variables(self, batch_size):
        # initialize h and c as zeros
        h_t = torch.zeros(batch_size, self.hidden_size).to(self.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return h_t, c_t


class BasicModel(AbstractModel):

    def __init__(self,
                 input_shape,
                 max_len,
                 hidden_size,
                 vocab_size,
                 device,
                 embedding_size=512):
        super(BasicModel, self).__init__(input_shape,
                                         max_len,
                                         hidden_size,
                                         vocab_size,
                                         device,
                                         embedding_size=embedding_size)
        self.decoder = BasicDecoder(hidden_size,
                                    embedding_size,
                                    vocab_size,
                                    device)


class BasicDecoder(nn.Module):

    def __init__(self, hidden_size, embedding_size, vocab_size, device):
        super(BasicDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm_cell = nn.LSTMCell(self.embedding_size * 2, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, states):
        input_img, input_w = x
        global_image, encoded_image = input_img
        h_tm1, c_tm1 = states

        embedded_w = self.embedding(input_w)
        x_t = torch.cat((global_image, embedded_w), dim=1)

        h_t, c_t = self.lstm_cell(x_t, (h_tm1, c_tm1))
        pt = self.output_layer(h_t)

        return pt, h_t, c_t

    def initialize_variables(self, batch_size):
        # initialize h and c as zeros
        h_t = torch.zeros(batch_size, self.hidden_size).to(self.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return h_t, c_t
