from pathlib import Path
import pandas as pd

ROOT_PATH = Path(__file__).absolute().parents[2]

"""
main file for creating splits of the training set where the number of images
are reduced and not the number of captions per image.

"""

prepped_path = ROOT_PATH.joinpath('data',
                                  'interim',
                                  'karpathy_split',
                                  'coco_sub_prep.csv')
# load prepped data
data_df = pd.read_csv(prepped_path)

# reduce the number of images until the total number of captions are roughly
# equal to the number of total captions of the reduced captions subsets.

