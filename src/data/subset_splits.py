import random as r
import pandas as pd
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parents[2]
SEED = 222

"""
Main file for creating the dataset splits, for the data exploration.


"""
prepped_path = ROOT_PATH.joinpath('data',
                                  'interim',
                                  'karpathy_split',
                                  'coco_sub_prep.csv')
# load prepped data
data_df = pd.read_csv(prepped_path)

# create initial subset with 1 caption per image

# create the next and so on adding an extra caption per image for each itr

