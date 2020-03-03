import random as r
import pandas as pd
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parents[2]
SEED = 222

"""
Main file for creating the dataset splits, for the data exploration.


"""


# load cleaned training dataset
coco_train_path = ROOT_PATH.joinpath('data',
                                     'interim',
                                     'karpathy_split',
                                     'coco_train_clean.csv')
coco_train_df = pd.read_csv(coco_train_path)
OG_SIZE = len(coco_train_df)

# remove images with less than 5 captions
# remove extra captions if some images have more than 5 captions
image_ids = list(set(coco_train_df.loc[:, 'image_id']))
# image_ids is a list so we do not introduce randomness from set
new_df = pd.DataFrame()
for image_id in image_ids:
    caps = coco_train_df.loc[coco_train_df['image_id'] == image_id, :]
    caps_size = len(caps)
    if caps_size == 5:
        # perfect amount of captions
        new_df = new_df.append(caps, ignore_index=True)
    elif caps_size > 5:
        # too many captions
        # remove extra captions
        tmp_df = pd.DataFrame().append(caps, ignore_index=True)
        new_df = new_df.append(tmp_df.loc[:5, :], ignore_index=True)

REMOVE_IMG_SIZE = len(new_df)
print('removed', OG_SIZE - REMOVE_IMG_SIZE, 'captions')  # removed 32 captions

# create initial subset with 1 caption per image

# create the next and so on adding an extra caption per image for each itr

