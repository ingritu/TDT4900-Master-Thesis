
from pathlib import Path
import numpy as np
from skimage.io import imread
from sklearn.utils import shuffle
import random as r
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

ROOT_PATH = Path(__file__).absolute().parents[2]


def data_generator(data_df, batch_size, step_per_epoch, image_path, seed=2222):
    """
    outputs data in batches
    """
    # TODO: order data to create batches with captions of roughly the same length
    r.seed(seed=seed)
    shuffle_state = r.randint(0, 10000)
    # infinite loop
    while True:
        # new Epoch have started
        # shuffle df
        data_df = shuffle(data_df, random_state=shuffle_state)
        data_df = data_df.reset_index()
        for step in range(step_per_epoch):
            # create a new batch
            batch_input = np.array([])
            batch_output = np.array([])
            for i in range(batch_size):
                image = get_image(image_path)
                label = get_label(data_df, i)
                batch_input = np.append(batch_input, image)
                batch_output = np.append(batch_output, label)
            # new shuffle state for next epoch
            shuffle_state = r.randint(0, 10000)
            yield [batch_input, batch_output]


def get_image(image_path):
    return imread(image_path)


def get_label(data_df, i):
    return np.zeros(5)  # placeholder



