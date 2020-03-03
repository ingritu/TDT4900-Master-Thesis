import torch


class Beam:

    def __init__(self,
                 image,
                 states,
                 beam_size,
                 input_token,
                 eos,
                 max_len,
                 vocab_size,
                 device,
                 beam_id=-1):
        # misc
        self.id = beam_id
        self.global_image, self.encoded_image = image
        self.h, self.c = states
        self.beam_size = beam_size
        self.num_unfinished = beam_size
        self.EOS = eos
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.device = device
        # initialize captions in beam
        # unfinished captions
        init_cap = torch.tensor(input_token).to(self.device)
        self.captions = init_cap.unsqueeze(0).expand(self.beam_size, -1)
        init_prev_w = torch.tensor(input_token).to(self.device)
        init_prev_w = init_prev_w.unsqueeze(0).expand(self.beam_size, -1)
        self.previous_words = init_prev_w.squeeze(1)
        self.top_scores = torch.zeros(self.beam_size, 1).to(self.device)
        # finished captions
        self.finished_caps = []  # these will never be sorted
        self.finished_caps_scores = []  # will never be sorted
        # Early stopping
        self.optimality_certificate = False

    def update(self, predictions, h, c):
        """
        Update Beam state with the new predictions. Opt out early if the
        optimality certificate is achieved before num_unfinished=0.

        Optimality certificate:
        When to Finish? Optimal Beam Search for Neural Text Generation
        (modulo beam size) (2017) by Huang et al.

        Parameters
        ----------
        predictions : torch.tensor. Predictions.
        h : torch.tensor. Current hidden states.
        c : torch.tensor. Current cell states.
        """
        # h, c: (n, num_unfinished, hidden_size)
        # predictions (num_unfinished, vocab_size)
        if self.num_unfinished == 0 or self.optimality_certificate:
            # Do nothing
            return

        # add probabilities
        # (beam_size, 1) --> (beam_size, vocab_size)
        scores = self.top_scores.expand_as(predictions)
        # (beam_size, v) + (num_unfinished, v) --> (num_unfinished, v)
        predictions = scores + predictions

        # flatten predictions and find top k
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
        finished_idx = list(set(range(len(next_word_idx))) -
                            set(unfinished_idx))

        # sort h, c, top_scores, previous_words
        # only keep the unfinished ones
        self.h = h[:, prev_word_idx[unfinished_idx]]
        self.c = c[:, prev_word_idx[unfinished_idx]]
        self.top_scores = top_probs[unfinished_idx].unsqueeze(1)
        self.previous_words = next_word_idx[unfinished_idx]

        # update finished captions
        beam_reduce_num = min(len(finished_idx), self.num_unfinished)
        if beam_reduce_num > 0:
            # add finished captions to finished
            self.finished_caps.extend(self.captions[finished_idx].tolist())
            self.finished_caps_scores.extend(top_probs[finished_idx])
            self.num_unfinished -= beam_reduce_num

            # find best complete caption
            best_fin_idx = self.find_best_comlpete_sequence()
            # check whether optimality certificate is achieved
            if len(self.finished_caps) < self.beam_size:
                # only check if there still are unfinished caps left
                # if there are none in unfinished then there is no
                # point to check whether the optimality certificate is obtained
                best_unfin_prob, _ = self.top_scores.topk(1, dim=0)
                self.optimality_certificate = \
                    self.finished_caps_scores[best_fin_idx] > best_unfin_prob

            if self.optimality_certificate:
                # no need for further calculations
                self.num_unfinished = 0

        # sort captions, only keep the unfinished ones
        # could not do this until after finished was updated
        self.captions = self.captions[unfinished_idx]

        if self.captions.size(0):
            # there are still more captions left
            # move captions to finished if length too long
            if self.captions.size(1) >= self.max_len:
                self.finished_caps.extend(self.captions.tolist())
                self.finished_caps_scores.extend(self.top_scores)
                self.captions = torch.empty(self.beam_size)
                self.num_unfinished = 0

    def find_best_comlpete_sequence(self):
        """
        Search the finished captions for the most probable caption.

        Returns
        -------
        The index of the most probable finished caption.
        """
        probs = torch.tensor(self.finished_caps_scores).to(self.device)
        _, idx = probs.topk(1, dim=0)
        return int(idx)

    def get_sequences(self):
        # only return unfinished
        return self.previous_words

    def get_encoded_image(self):
        return self.encoded_image.unsqueeze(0).expand(self.num_unfinished,
                                                      -1, -1)

    def get_global_image(self):
        return self.global_image.unsqueeze(0).expand(self.num_unfinished, -1)

    def get_hidden_states(self):
        return self.h

    def get_cell_states(self):
        return self.c

    def has_best_sequence(self):
        return len(self.finished_caps) == self.beam_size or \
               self.optimality_certificate

    def get_best_sequence(self):
        """
        Returns
        -------
        A list of sequence tokens.
        """
        assert self.has_best_sequence(), "Beam search incomplete"
        idx = self.find_best_comlpete_sequence()
        return self.finished_caps[idx]
