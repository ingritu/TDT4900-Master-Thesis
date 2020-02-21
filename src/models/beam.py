

class Beam:

    def __init__(self, image, input_token, beam_size, generator, beam_id=-1):
        self.id = beam_id
        self.encoded_image = image
        self.beam_size = beam_size
        self.num_unfinished = beam_size
        self.generator = generator
        # initialize captions in beam
        self.captions = [[input_token, 0.0]]
        self.caption_lengths = []
        self.unfinished_caps_idx = [i for i in range(beam_size)]

    def update(self, predictions):
        # preds (num_unfinished, vocab_size)
        pass

    def get_sequences(self):
        # only return unfinished, caps without endseq token
        return [c[0] for i, c in enumerate(self.captions)
                if i in set(self.unfinished_caps_idx)]

    def get_encoded_image(self):
        return self.encoded_image

    def get_items(self):
        seqs = self.get_sequences()
        images = [self.get_encoded_image() for _ in range(len(seqs))]
        return [images, seqs]

    def get_sequence_lengths(self):
        return self.caption_lengths
