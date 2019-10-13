from pathlib import Path
import numpy as np
from sklearn.utils import shuffle
import random as r
from src.data.load_vocabulary import load_vocabulary
from src.features.Resnet_features import load_visual_features
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

ROOT_PATH = Path(__file__).absolute().parents[2]


def data_generator(data_df, batch_size, steps_per_epoch,
                   wordtoix, feature_path, seed=2222):
    """
    outputs data in batches
    """
    # TODO: order data to create batches with captions of roughly the same length
    r.seed(seed)
    shuffle_state = r.randint(0, 10000)

    # load visual features
    visual_features = load_visual_features(feature_path)
    max_length = max([len(c.split()) for c in set(data_df.loc[:, 'clean_caption'])])
    vocab_size = len(wordtoix)
    # infinite loop
    while True:
        # new Epoch have started
        # shuffle df
        data_df = shuffle(data_df, random_state=shuffle_state)
        data_df = data_df.reset_index()
        for step in range(steps_per_epoch):
            # create a new batch
            x1 = np.array([])
            x2 = np.array([])
            y = np.array([])
            # Steps per epoch is equal to floor of
            # total_samples/batch_size
            # TODO: make steps per epoch equal to ceiling of
            # TODO (continued): total_samples/batch_size
            for i in range(batch_size * step, batch_size * (step + 1)):
                image = get_image(visual_features, data_df, i)
                caption = get_caption(data_df, i)
                # create partial captions
                seq = [wordtoix[word] for word in caption.split(' ')
                       if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for j in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:j], seq[j]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    x1 = np.append(x1, image)
                    x2 = np.append(x2, in_seq)
                    y = np.append(y, out_seq)
            # new shuffle state for next epoch
            shuffle_state = r.randint(0, 10000)
            yield [[x1, x2], y]


def get_image(visual_features, data_df, i):
    image_id = data_df.loc[i, 'image_id']
    return visual_features[image_id]


def get_caption(data_df, i):
    return data_df.loc[i, 'clean_caption']
