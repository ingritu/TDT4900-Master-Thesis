from copy import deepcopy
import torch


class Beam:

    def __init__(self,
                 image,
                 beam_size,
                 input_token,
                 eos,
                 max_len,
                 vocab_size,
                 device,
                 beam_id=-1):
        self.id = beam_id
        self.global_image, self.encoded_image = image
        self.beam_size = beam_size
        self.num_unfinished = beam_size
        self.EOS = eos
        self.max_len = max_len
        self.vocab_size = vocab_size
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
        # Early stopping
        self.optimality_certificate = False

    def update(self, predictions, h, c):
        # TODO: implement early stopping on optimality certificate achieved
        # h, c: (n, num_unfinished, hidden_size)
        # predictions (num_unfinished, vocab_size)
        if self.num_unfinished == 0 or self.optimality_certificate:
            # Do nothing
            return

        # add probabilities
        predictions = self.top_scores.expand_as(predictions) + predictions

        top_probs, top_words = predictions.view(-1).topk(self.num_unfinished,
                                                         dim=0,
                                                         largest=True,
                                                         sorted=True)
        # top_probs, top_words: (num_unfinished)
        prev_word_idx = top_words / self.vocab_size  # previous index
        next_word_idx = top_words % self.vocab_size  # word predicted

        # add predicted words to caption
        self.captions = torch.cat([self.captions[prev_word_idx],
                                   next_word_idx.unsqueeze(1)],
                                  dim=1)
        unfinished_idx = [idx for idx, next_word in enumerate(next_word_idx)
                          if next_word != self.EOS]
        finished_idx = [set(range(len(next_word_idx))) - set(unfinished_idx)]

        beam_reduce_num = min(len(finished_idx), self.num_unfinished)
        if beam_reduce_num > 0:
            # add finished captions to finished
            self.finished_caps.extend(self.captions[finished_idx].tolist())
            self.finished_caps_scores.extend(top_probs[finished_idx])
            self.num_unfinished -= beam_reduce_num

        # sort captions, h, c, top_scores, previous_words
        self.captions = self.captions[unfinished_idx]
        h = h[:, prev_word_idx[unfinished_idx]]
        c = c[:, prev_word_idx[unfinished_idx]]
        self.top_scores = self.top_scores[unfinished_idx].unsqueeze(1)
        self.previous_words = next_word_idx[unfinished_idx].unsqueeze(1)

        # move captions to finished if length too long
        if max(map(len, self.captions)) >= self.max_len:
            self.finished_caps = self.finished_caps + self.captions
            self.captions = []
            self.num_unfinished = 0

        return h, c

    def get_sequences(self):
        """
        Returns
        -------
        list of the most recently predicted words. word is EOS
        if the caption is technically finished.
        """
        # only return unfinished, and add zeroes for finished captions
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
