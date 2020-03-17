import random as r
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT_PATH = Path(__file__).absolute().parents[2]
SEED = 222

"""
Main file for creating the dataset splits, for the data exploration.
C1 < C2 < C3 < C4 < C5
"""
if __name__ == '__main__':
    prepped_path = ROOT_PATH.joinpath('data',
                                      'interim',
                                      'karpathy_split',
                                      'coco_sub_prep.csv')
    # load prepped data
    data_df = pd.read_csv(prepped_path)

    # seed random
    r.seed(SEED)

    subsets = [defaultdict(list) for _ in range(5)]
    subset_ds = [defaultdict(list) for _ in range(5)]

    image_ids = list(set(data_df.loc[:, 'image_id']))
    cols = list(data_df.columns)

    counter = 0
    for image_id in image_ids:
        # create pick order
        indices = [i for i in range(5)]
        r.shuffle(indices)
        for c_idx in range(len(subsets)):
            subsets[c_idx][image_id] = indices[: c_idx + 1]

        # put data into dictionaries
        caps = data_df.loc[data_df['image_id'] == image_id, :]
        caps = caps.reset_index(drop=True)
        for i, sub in enumerate(subsets):
            for idx in sub[image_id]:
                for col in cols:
                    subset_ds[i][col].append(caps.loc[idx, col])
        counter += 1
        if counter % 1000 == 0:
            print(counter)

    C1_df = pd.DataFrame(subset_ds[0])
    C2_df = pd.DataFrame(subset_ds[1])
    C3_df = pd.DataFrame(subset_ds[2])
    C4_df = pd.DataFrame(subset_ds[3])
    C5_df = pd.DataFrame(subset_ds[4])

    processed_path = ROOT_PATH.joinpath('data',
                                        'processed')
    sub_C_dir = processed_path.joinpath('cap_subsets')

    if not sub_C_dir.is_dir():
        sub_C_dir.mkdir(parents=True)

    C1_df.to_csv(sub_C_dir.joinpath('c1.csv'))
    C2_df.to_csv(sub_C_dir.joinpath('c2.csv'))
    C3_df.to_csv(sub_C_dir.joinpath('c3.csv'))
    C4_df.to_csv(sub_C_dir.joinpath('c4.csv'))
    C5_df.to_csv(sub_C_dir.joinpath('c5.csv'))
