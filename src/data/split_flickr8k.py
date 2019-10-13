from pathlib import Path
import pandas as pd

ROOT_PATH = Path(__file__).absolute().parents[2]


def make_train_val_test_split(df_path, split_paths, save_path):
    # get df
    cap_df = pd.read_csv(df_path)
    train_path = split_paths[0]
    val_path = split_paths[1]
    test_path = split_paths[2]

    # read train val test files
    with open(train_path, 'r') as train_file:
        train_images = [im_id.strip() for im_id in train_file.readlines()
                        if len(im_id) > 0]
    with open(val_path, 'r') as val_file:
        val_images = [im_id.strip() for im_id in val_file.readlines()
                      if len(im_id) > 0]
    with open(test_path, 'r') as test_file:
        test_images = [im_id.strip() for im_id in test_file.readlines()
                       if len(im_id) > 0]
    train_df = cap_df.loc[cap_df.loc[:, 'image_id'].isin(train_images), :]
    val_df = cap_df.loc[cap_df.loc[:, 'image_id'].isin(val_images), :]
    test_df = cap_df.loc[cap_df.loc[:, 'image_id'].isin(test_images), :]
    # make a merged version
    full_df = train_df.copy()
    full_df = full_df.append(val_df)
    full_df = full_df.append(test_df)

    # save splits
    train_df.to_csv(save_path.joinpath('Flickr8k_train.csv'))
    val_df.to_csv(save_path.joinpath('Flickr8k_val.csv'))
    test_df.to_csv(save_path.joinpath('Flickr8k_test.csv'))
    # save full set
    full_df.to_csv(save_path.joinpath('Flickr8k_full.csv'))
    print("Finished making splits!")
