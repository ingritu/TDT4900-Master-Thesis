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
    # infinite loop
    while True:
        # new Epoch have started
        # new shuffle state for next epoch
        shuffle_state = r.randint(0, 10000)
        # shuffle df
        data_df = shuffle(data_df, random_state=shuffle_state)
        data_df = data_df.reset_index(drop=True)
        for step in range(steps_per_epoch):
            # create a new batch
            x1 = []
            x2 = []
            caption_lengths = []
            # Steps per epoch is equal to floor of
            # total_samples/batch_size
            # TODO: make steps per epoch equal to ceiling of
            # TODO (continued): total_samples/batch_size
            for i in range(batch_size * step, batch_size * (step + 1)):
                image = get_image(features, data_df, i)
                caption = get_caption(data_df, i)
                # create caption number sequence
                seq = [wordtoix[word] for word in caption.split(' ')
                       if word in wordtoix]
                caption_lengths.append(len(seq))

                x1.append(image)
                x2.append(torch.tensor(seq))

            # pad input sequence
            x2 = pad_sequences(x2, max_length)  # output is a tensor

            x1 = torch.tensor(x1)  # convert to tensor
            caption_lengths = np.array(caption_lengths)  # convert to array
            yield [[x1, x2], caption_lengths]


def get_image(visual_features, data_df, i):
    """
    Get image encoding.

    Parameters
    ----------
    visual_features : dict.
    data_df : DataFrame.
    i : int. index.

    Returns
    -------
    Image encoding.
    """
    image_name = data_df.loc[i, 'image_name']
    return visual_features[image_name]


def get_caption(data_df, i):
    """
    Get caption string.
    Parameters
    ----------
    data_df : DataFrame.
    i : int. Index.

    Returns
    -------
    Caption string.
    """
    return data_df.loc[i, 'clean_caption']


def pad_sequences(sequences, maxlen):
    """
    Pad sequences.

    Parameters
    ----------
    sequences : list. list of tensors.
    maxlen : int. Size to pad to.

    Returns
    -------
    Padded sequences as one tensor.
    """
    num = len(sequences)
    out_dims = (num, maxlen)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
    return out_tensor

