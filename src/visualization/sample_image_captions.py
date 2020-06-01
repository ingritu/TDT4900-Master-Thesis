from pathlib import Path
import json
import pandas as pd
from sklearn.utils import shuffle
import numpy as np


ROOT_PATH = Path(__file__).absolute().parents[2]


def get_image_caption(model_path, image_id):
    print('get image caption')
    with open(model_path.joinpath('TEST_test_result.json')) as json_file:
        result = json.load(json_file)

    # result is a list of dictionaries
    for obj in result:
        if obj['image_id'] == image_id:
            return result['caption']
    return None


def get_model_type(model_name, exp=True):
    print('get model type')
    if model_name[0] == 'a':
        return 'adaptive' if not exp else 'adaptive_exp'
    return 'basic' if not exp else 'basic_exp'


def get_model_trainset(model_path, model_type):
    print('get model trainset')
    with open(model_path.joinpath(model_type + "_log.txt")) as log_file:
        data = log_file.readlines()
    data = data[3]  # line where training data is logged
    data = data.split(' ')[-1]  # dataset path is the last word at the line
    path_length = len(data)
    reversed_data = data[::-1]
    idx = reversed_data.find('/')
    data = data[path_length - idx:]
    data = data.replace('_train_clean.csv', '')
    return data


def sample_image_captions(models, test_path, res_df, seed, sample_size=20):
    """
    Fetches the captions that models produced on a subset of the test set.

    Parameters
    ----------
    models : list.
        List of model names.
    test_path : Path or str.
        Path to the test set.
    res_df : DataFrame.
        DataFrame where data is saved.
    seed : int.
        Random seed.
    sample_size : int.
        Size of sample set. Default is 20.

    Returns
    -------
    res_df : DataFrame.
        Updated res_df.
    """
    np.random.seed(seed)  # shuffle is seeded through numpy
    models_dir = ROOT_PATH.joinpath('models')

    # pick images to fetch captions for
    test_df = pd.read_csv(test_path)
    shuff = shuffle(test_df)
    shuff = shuff.reset_index(drop=True)
    sample_df = shuff.loc[:sample_size-1, ['image_id', 'image_name']]

    index = len(res_df)

    for model_name in models:
        model_type = get_model_type(model_name)
        model_path = models_dir.joinpath(model_name)
        trainset = get_model_trainset(model_path, model_type)

        for i in range(len(sample_df)):
            image_id = sample_df.loc[i, 'image_id']
            image_name = sample_df.loc[i, 'image_name']
            caption = get_image_caption(model_path, image_id)

            if caption is not None:
                # add to df
                res_df.loc[index, 'model_name'] = model_name
                res_df.loc[index, 'model_type'] = model_type
                res_df.loc[index, 'dataset'] = trainset
                res_df.loc[index, 'image_id'] = image_id
                res_df.loc[index, 'image_name'] = image_name
                res_df.loc[index, 'caption'] = caption

                index += 1

    res_df = res_df.drop_duplicates(ignore_index=True)
    return res_df
