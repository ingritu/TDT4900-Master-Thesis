from pathlib import Path
import pandas as pd
import numpy as np

import argparse

ROOT_PATH = Path(__file__).absolute().parents[2]

MAX_CAPS_IN_FILE = 8000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='c1',
                        help='subset to prepare to feed to google translate.')
    args = vars(parser.parse_args())

    subset_ = args['subset']

    external_path = ROOT_PATH.joinpath('data', 'external')
    processed_path = ROOT_PATH.joinpath('data', 'processed')

    save_path = external_path.joinpath(subset_)
    if not save_path.is_dir():
        save_path.mkdir(parents=True)

    df = pd.read_csv(processed_path.joinpath('cap_subsets',
                                             subset_ + '.csv'))
    df_size = len(df)
    files = int(np.ceil(df_size / MAX_CAPS_IN_FILE))
    print("number of new files", files)
    start_idx = 0
    for idx in range(files):
        end_idx = min((idx + 1) * MAX_CAPS_IN_FILE, df_size)
        sub_i = df.loc[start_idx: end_idx - 1, ['image_name', 'image_id',
                                                'caption_id', 'caption',
                                                'clean_caption']]
        sub_i.to_csv(save_path.joinpath(subset_ + '_' + str(idx) + '.txt'))
        start_idx = end_idx
