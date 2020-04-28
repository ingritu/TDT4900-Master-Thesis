from pathlib import Path
import pandas as pd
from collections import defaultdict
from shutil import copy
import random as r

from src.data.data_cleaning import basic_data_cleaning


ROOT_PATH = Path(__file__).absolute().parents[2]
SEED = 222  # do not switch this seed


def subset_prep(dataset_path, out_path):
    coco_train_df = pd.read_csv(dataset_path)
    og_size = len(coco_train_df)

    # remove images with less than 5 captions
    # remove extra captions if some images have more than 5 captions
    image_ids = list(set(coco_train_df.loc[:, 'image_id']))
    print('Number of images in full set', len(image_ids))
    # image_ids is a list so we do not introduce randomness from set
    new_d = defaultdict(list)
    cols = list(coco_train_df.columns)
    index = 0
    few = 0
    for image_id in image_ids:
        caps = coco_train_df.loc[coco_train_df['image_id'] == image_id, :]
        caps = caps.reset_index(drop=True)
        caps_size = len(caps)
        if caps_size >= 5:
            # perfect amount of captions or too many
            for col in cols:
                for i in range(5):
                    new_d[col].append(caps.loc[i, col])
        else:
            few += caps_size
        index += 1
        if index % 1000 == 0:
            print(index)

    new_df = pd.DataFrame(new_d)
    remove_img_size = len(new_df)
    print('removed', og_size - remove_img_size, 'captions')
    print('few captions', few)
    # these numbers are dependent on how rigorous the initial cleaning was,
    # in particular the unk-percentage, which may remove captions.
    # removed 32 captions as a result of not enough caps
    # total removed caps are 344 unk=0.4
    # 432 unk=0.3

    # save to file
    new_df.to_csv(out_path)


def make_caption_subsets(prep_path, out_path):
    # load prepped data
    data_df = pd.read_csv(prep_path)

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

    c1_df = pd.DataFrame(subset_ds[0])
    c2_df = pd.DataFrame(subset_ds[1])
    c3_df = pd.DataFrame(subset_ds[2])
    c4_df = pd.DataFrame(subset_ds[3])
    c5_df = pd.DataFrame(subset_ds[4])

    if not out_path.is_dir():
        out_path.mkdir(parents=True)

    c1_df.to_csv(out_path.joinpath('c1.csv'))
    c2_df.to_csv(out_path.joinpath('c2.csv'))
    c3_df.to_csv(out_path.joinpath('c3.csv'))
    c4_df.to_csv(out_path.joinpath('c4.csv'))
    c5_df.to_csv(out_path.joinpath('c5.csv'))


def make_para_subs(para_path, save_dir):
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

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)

    for i, df in enumerate(para_dfs):
        df.to_csv(save_dir.joinpath('p' + str(i + 1) + '.csv'))
    # save p5 to interim
    copy(para_path, save_dir.joinpath('p5.csv'))
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

    for i, p_df in enumerate(para_dfs):  # 6 to 10
        num = str(i + 6)
        combo_df = c5_df.append(p_df, ignore_index=True)
        combo_df.to_csv(para_path.joinpath('cp' + num + '.csv'))

    # c1+p1, c2+p2, c3+p3, c4+p4
    # load the rest of the subsets
    c1_to_4_dfs = [pd.read_csv(sub_path.joinpath('c' + str(i) + '.csv'))
                   for i in range(1, 5)]

    for i, (c_df, p_df) in enumerate(zip(c1_to_4_dfs, para_dfs)):
        num = str(i + 1)
        combo_df = c_df.append(p_df, ignore_index=True)
        combo_df.to_csv(para_path.joinpath('c' + num + 'p' + num + '.csv'))


if __name__ == '__main__':
    r.seed(SEED)

    external_path = ROOT_PATH.joinpath('data', 'external')
    processed_path = ROOT_PATH.joinpath('data', 'processed')
    interim_path = ROOT_PATH.joinpath('data', 'interim')

    sub_path_ = interim_path.joinpath('cap_subsets')
    para_path_ = external_path.joinpath('p5.csv')
    para_dir_ = interim_path.joinpath('paraphrases')

    dataset_path_ = interim_path.joinpath('karpathy_split',
                                          'coco_train_clean.csv')
    prepped_path = ROOT_PATH.joinpath('data',
                                      'interim',
                                      'karpathy_split',
                                      'coco_sub_prep.csv')
    val_path = interim_path.joinpath('karpathy_split',
                                     'coco_val.csv')
    # move val set to processed
    copy(val_path, processed_path.joinpath('karpathy_split', 'coco_val.csv'))

    subset_prep(dataset_path_, prepped_path)
    make_caption_subsets(prepped_path, sub_path_)
    make_para_subs(para_path_, para_dir_)
    make_combo_files(sub_path_, para_dir_)

    # clean the datasets
    print('Cleaning ...')
    base_str = str(para_dir_)
    for dir_ in [para_dir_, sub_path_]:
        base_str = str(dir_)
        for df_path_ in dir_.glob('*.csv'):
            filename = str(df_path_)[len(base_str) + 1:-4]
            print(filename)
            save_path_ = processed_path.joinpath('karpathy_split',
                                                 filename +
                                                 '_train_clean.csv')
            voc_path_ = processed_path.joinpath('karpathy_split',
                                                filename + '_vocabulary.csv')
            basic_data_cleaning(df_path_, save_path_, voc_path_,
                                threshold=5,
                                cutoff_value=16,
                                unk_percentage=2.0)
