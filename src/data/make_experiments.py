from pathlib import Path
import pandas as pd
from collections import defaultdict
from shutil import copy

ROOT_PATH = Path(__file__).absolute().parents[2]

external_path = ROOT_PATH.joinpath('data', 'external')
processed_path = ROOT_PATH.joinpath('data', 'processed')
interim_path = ROOT_PATH.joinpath('data', 'interim')


def make_para_subs(para_path):
    d = {key: defaultdict(list) for key in range(1, 5)}
    p5_df = pd.read_csv(para_path)
    para_count = len(p5_df)
    for i in range(0, para_count, 5):
        if i % 1000 == 0:
            print(i)
        image_ids = list(p5_df.loc[i: i + 4, 'image_id'].to_numpy())
        image_names = list(p5_df.loc[i: i + 4, 'image_name'].to_numpy())
        caption_ids = list(p5_df.loc[i: i + 4, 'caption_id'].to_numpy())
        captions = list(p5_df.loc[i: i + 4, 'caption'].to_numpy())
        for key in d:
            d[key]['image_id'].extend(image_ids[:key])
            d[key]['image_name'].extend(image_names[:key])
            d[key]['caption_id'].extend(caption_ids[:key])
            d[key]['caption'].extend(captions[:key])
    para_dfs = [pd.DataFrame(data=d[key]) for key in d]
    for i, df in enumerate(para_dfs):
        df.to_csv(interim_path.joinpath('p' + str(i + 1) + '.csv'))
    # save p5 to interim
    copy(para_path, interim_path.joinpath('p5.csv'))
    # all paraphrase files are now in interim


def make_combo_files(sub_path, para_path):
    # make combo files
    # c5+p1, +p2, +p3, +p4, +p5; call them cp6, ..., cp10
    # load c5
    c5_df = pd.read_csv(sub_path.joinpath('c5.csv'))
    c5_df = c5_df.loc[:, ['image_id', 'image_name', 'caption_id', 'caption']]
    # load para_dfs
    para_dfs = [pd.read_csv(para_path.joinpath('p' + str(i) + '.csv'))
                for i in range(1, 6)]
    cp6_to_10_dfs = []
    for p_df in para_dfs:  # 6 to 10
        cp6_to_10_dfs.append(c5_df.append(p_df, ignore_index=True))
    # save to interim
    for i, combo_df in enumerate(cp6_to_10_dfs):
        num = i + 6
        combo_df.to_csv(interim_path.joinpath('cp' + str(num) + '.csv'))

    # c1+p1, c2+p2, c3+p3, c4+p4
    # load the rest of the subsets
    c1_to_4_dfs = [pd.read_csv(sub_path.joinpath('c' + str(i) + '.csv'))
                   for i in range(1, 5)]
    cp_double_dfs = []
    for c_df, p_df in zip(c1_to_4_dfs, para_dfs):
        cp_double_dfs.append(c_df.append(p_df, ignore_index=True))
    # save to interim
    for i, combo_df in enumerate(cp_double_dfs):
        num = str(i + 1)
        combo_df.to_csv(interim_path.joinpath('c' + num + 'p' + num + '.csv'))


if __name__ == '__main__':
    sub_path_ = interim_path.joinpath('cap_subsets')
    para_path_ = external_path.joinpath('p5.csv')

    make_para_subs(para_path_)
    make_combo_files(sub_path_, interim_path)

























