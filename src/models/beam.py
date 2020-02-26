from torch.nn.utils.rnn import pack_padded_sequence
from copy import deepcopy
import numpy as np
import torch


class Beam:

    def __init__(self,
                 image,
                 beam_size,
                 input_token,
                 eos,
                 max_len,
                 device,
                 beam_id=-1):
        self.id = beam_id
        self.global_image, self.encoded_image = image
        self.beam_size = beam_size
        self.num_unfinished = beam_size
        self.EOS = eos
        self.max_len = max_len
        self.device = device
        # initialize captions in beam
        # unfinished captions
        self.captions = torch.tensor(input_token).to(self.device)
        self.captions = self.captions.unsqueeze(0).expand(self.beam_size, -1)
        self.previous_words = torch.tensor(input_token).to(self.device)
        self.previous_words = \
            self.previous_words.unsqueeze(0).expand(self.beam_size, -1)

        self.select_index = torch.zeros(self.beam_size).to(self.device)
        self.top_scores = torch.zeros(self.beam_size, 1).to(self.device)
        # finished captions
        self.finished_caps = []
        self.finished_caps_scores = []
        self.longest_length = max([c[0].size() for c in self.captions])

    def update(self, predictions, h, c):
        # h, c: (n, num_unfinished, hidden_size)
        # predictions (num_unfinished, vocab_size)
        if self.num_unfinished == 0:
            # Do nothing
            return

        top_probs, top_words = predictions.view(-1).topk(self.num_unfinished,
                                                         dim=0,
                                                         largest=True,
                                                         sorted=True)
        print('probs', top_probs.size())
        print('words', top_words.size())













        tmp_caps = []
        for caption, preds in zip(self.captions, predictions):
            # preds (vocab_size)
            # expand each caption


            for word, prob in zip(top_words, top_probs):
                new_partial_cap = deepcopy(caption[0])
                new_partial_cap.append(word)
                new_partial_cap_prob = caption[1] + preds[word]

                if word == self.EOS:
                    # add to finished if en token
                    self.finished_caps.append([new_partial_cap,
                                               new_partial_cap_prob])
                    self.num_unfinished -= 1
                else:
                    # add cap and prob to tmp list
                    tmp_caps.append([new_partial_cap,
                                     new_partial_cap_prob])

        # update unfinished captions
        self.captions = tmp_caps
        self.captions.sort(key=lambda l: l[1])
        self.captions = self.captions[-self.num_unfinished:]
        # update caption_lengths
        self.caption_lengths = [c[0].size() for c in self.captions]
        #print(self.caption_lengths)

        # move captions to finished if length too long
        if self.caption_lengths[0] >= self.max_len:
            self.finished_caps = self.finished_caps + self.captions
            self.captions = []
            self.num_unfinished = 0

    def get_sequences(self):
        """
        Returns
        -------
        list of the most recently predicted words. word is EOS
        if the caption is technically finished.
        """
        # only return unfinished, and add zeroes for finished captions
        print('get_sequence', self.previous_words.size())
        return self.previous_words.squeeze(1)

    def get_encoded_image(self):
        return self.encoded_image

    def get_global_image(self):
        return self.global_image

    def get_image_features(self):
        # consider just keeping this in memory to avoid extra computation
        enc_images = [self.encoded_image for _ in range(self.beam_size)]
        global_images = [self.global_image for _ in range(self.beam_size)]
        return global_images, enc_images

    def get_items(self):
        seqs = self.get_sequences()
        global_images, enc_images = self.get_image_features()
        cap_lens = self.get_sequence_lengths()
        return global_images, enc_images, seqs, cap_lens

    def get_sequence_lengths(self):
        return self.caption_lengths

    def has_best_sequence(self):
        return len(self.finished_caps) == self.beam_size

    def get_best_sequence(self):
        assert self.has_best_sequence(), "Beam search incomplete"
        self.finished_caps.sort(key=lambda l: l[1])
        out = self.finished_caps[-1][0]
        return out
