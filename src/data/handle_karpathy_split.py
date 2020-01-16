import json
from pathlib import Path
import pandas as pd

ROOT_PATH = Path(__file__).absolute().parents[2]


def order_raw_data_and_move_to_interim(data_path, dataset):
    with open(data_path, 'r') as json_file:
        data_dict = json.load(json_file)

    train_dict = {}
    test_dict = {}
    val_dict = {}

    columns = ['image_id', 'caption_id', 'caption']

    for col in columns:
        train_dict[col] = []
        test_dict[col] = []
        val_dict[col] = []
    full_dict = {'train': train_dict, 'val': val_dict, 'test': test_dict}

    images = data_dict['images']

    for image in images:
        # get image name
        image_id = image['filename']
        # get split
        split = image['split']
        # go through captions and add them to dict split
        captions = image['sentences']
        for caption in captions:
            raw_caption = caption['raw']
            caption_id = image_id[:-4] + '#' + str(caption['sentid'])
            # add info to dicts
            if split != 'restval':
                full_dict[split]['image_id'].append(image_id)
                full_dict[split]['caption_id'].append(caption_id)
                full_dict[split]['caption'].append(raw_caption)
            else:
                # supplement training dataset with restval
                full_dict['train']['image_id'].append(image_id)
                full_dict['train']['caption_id'].append(caption_id)
                full_dict['train']['caption'].append(raw_caption)

    train_df = pd.DataFrame(train_dict, columns=columns)
    test_df = pd.DataFrame(test_dict, columns=columns)
    val_df = pd.DataFrame(val_dict, columns=columns)

    save_path = ROOT_PATH.joinpath('data', 'interim', 'karpathy_split')

    train_file = save_path.joinpath(dataset + '_train.csv')
    test_file = save_path.joinpath(dataset + '_test.csv')
    val_file = save_path.joinpath(dataset + '_val.csv')

    train_df.to_csv(train_file)
    test_df.to_csv(test_file)
    val_df.to_csv(val_file)
    print("finished job!!")
