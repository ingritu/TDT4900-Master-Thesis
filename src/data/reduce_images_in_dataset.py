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
sub_c_dir = ROOT_PATH.joinpath('data',
                               'processed',
                               'cap_subsets')
C1_size = len(pd.read_csv(sub_c_dir.joinpath('c1.csv')))
C2_size = len(pd.read_csv(sub_c_dir.joinpath('c2.csv')))
C3_size = len(pd.read_csv(sub_c_dir.joinpath('c3.csv')))
C4_size = len(pd.read_csv(sub_c_dir.joinpath('c4.csv')))
C5_size = len(pd.read_csv(sub_c_dir.joinpath('c5.csv')))
print('c1', C1_size, 'c2', C2_size,
      'c3', C3_size, 'c4', C4_size,
      'c5', C5_size)
# reduce the number of images until the total number of captions are roughly
# equal to the number of total captions of the reduced captions subsets.

