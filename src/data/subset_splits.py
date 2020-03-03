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

# seed random
r.seed(SEED)
d = {}

image_ids = list(set(data_df.loc[:, 'image_id']))

# create pick order
for image_id in image_ids:
    indices = [i for i in range(5)]
    r.shuffle(indices)
    d[image_id] = indices
print(d[123277])

C1 = {}
C2 = {}
C3 = {}
C4 = {}
C5 = {}

# create initial subset with 1 caption per image

# create the next and so on adding an extra caption per image for each itr

