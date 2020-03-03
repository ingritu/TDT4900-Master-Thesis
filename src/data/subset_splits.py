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

# remove images with less than 5 captions

# remove extra captions if some images have more than 5 captions

# create initial subset with 1 caption per image

# create the next and so on adding an extra caption per image for each itr

