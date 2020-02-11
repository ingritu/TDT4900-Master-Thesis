import torch
from pathlib import Path
import numpy as np
from sklearn.utils import shuffle
import random as r

ROOT_PATH = Path(__file__).absolute().parents[2]


def data_generator(data_df, batch_size, steps_per_epoch,
                   wordtoix, features, seed=2222):
    """
    outputs data in batches
    """
    # TODO: order data to create batches with captions
    #  of roughly the same length
    r.seed(seed)

    # load visual features
    max_length = max([len(c.split())
                      for c in set(data_df.loc[:, 'clean_caption'])])
    vocab_size = len(wordtoix)
    # infinite loop
    while True:
        # new Epoch have started
        # new shuffle state for next epoch
        shuffle_state = r.randint(0, 10000)
        # shuffle df
        data_df = shuffle(data_df, random_state=shuffle_state)
        data_df = data_df.reset_index()
        for step in range(steps_per_epoch):
            # create a new batch
            x1 = []
            x2 = []
            y = []
            caption_lengths = []
            # Steps per epoch is equal to floor of
            # total_samples/batch_size
            # TODO: make steps per epoch equal to ceiling of
            # TODO (continued): total_samples/batch_size
            for i in range(batch_size * step, batch_size * (step + 1)):
                image = get_image(features, data_df, i)
                caption = get_caption(data_df, i)
                # create partial captions
                seq = [wordtoix[word] for word in caption.split(' ')
                       if word in wordtoix]
                caption_lengths.append(len(seq))
                # split one sequence into multiple X, y pairs
                for j in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:j], seq[j]
                    # convert in_seq to tensor because pad_sequence expects it
                    in_seq = torch.tensor(in_seq)
                    # store
                    x1.append(image)
                    x2.append(in_seq)
                    y.append(out_seq)

            # pad input sequence
            x2 = pad_sequences(x2, max_length)  # output is a tensor

            # encode output sequence
            y = np.array(y)
            # y = to_categorical(y, num_classes=vocab_size, dtype=np.int_)

            x1 = torch.tensor(x1)  # convert to tensor
            # print(y)
            y = torch.from_numpy(y)  # convert to tensor
            yield [[x1, x2], y, caption_lengths]


def get_image(visual_features, data_df, i):
    image_id = data_df.loc[i, 'image_id']
    return visual_features[image_id]


def get_caption(data_df, i):
    return data_df.loc[i, 'clean_caption']


def to_categorical(y, num_classes=None, dtype='float32'):
    # copied from keras for convenience
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def pad_sequences(sequences, maxlen):
    num = len(sequences)
    out_dims = (num, maxlen)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
    return out_tensor

