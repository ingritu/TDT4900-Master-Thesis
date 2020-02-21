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
                 beam_id=-1):
        self.id = beam_id
        self.encoded_image = image
        self.beam_size = beam_size
        self.num_unfinished = beam_size
        self.EOS = eos
        self.max_len = max_len
        # initialize captions in beam
        self.captions = [[input_token, 0.0]]  # unfinished
        self.finished_caps = []
        self.caption_lengths = [len(c[0]) for c in self.captions]

    def update(self, predictions, decoding_lengths):
        # preds (num_unfinished, maxlen, vocab_size)
        # decoding_lengths (num_unfinished, 1)
        if self.num_unfinished == 0:
            # Do nothing
            return

        # pack padded sequences to get them down to real length
        predictions = pack_padded_sequence(predictions,
                                           decoding_lengths,
                                           batch_first=True)[0]
        tmp_caps = []
        for caption, preds in zip(self.captions, predictions):
            # idx is the caption index of a caption without endseq
            # expand each caption
            preds = preds.detach().numpy()[-1]
            words_predicted = np.argsort(preds)[-self.beam_size:]

            for word in words_predicted:
                new_partial_cap = deepcopy(caption[0])
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
        self.caption_lengths = [len(c[0]) for c in self.captions]

        # move captions to finished if length too long
        if self.caption_lengths[0] >= self.max_len:
            self.finished_caps = self.finished_caps + self.captions
            self.captions = []
            self.num_unfinished = 0

    def get_sequences(self):
        # only return unfinished, caps without endseq token
        return [torch.tensor(c[0]) for c in self.captions]

    def get_encoded_image(self):
        return self.encoded_image

    def get_items(self):
        seqs = self.get_sequences()
        images = [self.get_encoded_image() for _ in range(len(seqs))]
        cap_lens = self.get_sequence_lengths()
        assert len(seqs) == len(images) and len(images) == len(cap_lens), \
            "Something is wrong. Output sizes does not match"
        return images, seqs, cap_lens

    def get_sequence_lengths(self):
        return self.caption_lengths

    def has_best_sequence(self):
        return len(self.finished_caps) == self.beam_size

    def get_best_sequence(self):
        assert self.has_best_sequence(), "Beam search incomplete"
        self.finished_caps.sort(key=lambda l: l[1])
        return self.finished_caps[-1]
