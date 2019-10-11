
from pathlib import Path
import numpy as np
from skimage.io import imread
import random as r

ROOT_PATH = Path(__file__).absolute().parents[2]


def data_generator(data_df, batch_size, epochs, image_path, seed=2222):
    """
    outputs data in batches
    """
    r.seed(seed=seed)
    shuffle_state = r.randint(0, 10000)
    # infinite loop
    while True:

        for ep in range(epochs):
            batch_input = np.array([])
            batch_output = np.array([])
            # shuffle df
            data_df.shuffle(random_state=shuffle_state)

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



