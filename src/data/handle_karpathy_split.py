import json
from pathlib import Path
from copy import deepcopy
import pandas as pd

ROOT_PATH = Path(__file__).absolute().parents[2]


def order_raw_data_and_move_to_interim(data_path, dataset, ann_path):
    data_path = Path(data_path)
    ann_path = Path(ann_path)
    with open(data_path, 'r') as json_file:
        data_dict = json.load(json_file)

    train_dict = {}
    test_dict = {}
    val_dict = {}

    columns = ['image_id', 'image_name', 'caption_id', 'caption']

    for col in columns:
        train_dict[col] = []
        test_dict[col] = []
        val_dict[col] = []
    full_dict = {'train': train_dict, 'val': val_dict, 'test': test_dict}

    init_dict = {
        "info": {
            "description": "description",
            "url": "www",
            "version": 1.0,
            "year": 2020,
            "contributor": "contributor",
            "date_created": "yyyy-mm-dd hh:mm:ss"
            },
        "images": [],
        "licenses": [],
        "type": "caption",
        "annotations": []
    }

    ann_dict = {
        "test": {},
        "val": {},
        "train": {}
    }

    for key in ann_dict:
        ann_dict[key] = deepcopy(init_dict)

    ann_images = {
        "test": [],
        "val": [],
        "train": []
    }
    ann_annotations = {
        "test": [],
        "val": [],
        "train": []
    }

    images = data_dict['images']

    for image in images:
        ann_image_obj = {
            "license": 0,
            "url": "www",
            "width": 0,
            "height": 0,
            "date_captured": "yyyy-mm-dd hh:mm:ss"
        }

        # get image name
        image_name = image['filename']
        image_id = image['imgid']

        # add info to ann_image_obj
        ann_image_obj['file_name'] = image_name
        ann_image_obj['id'] = image_id

        # get split
        split = image['split']
        # save ann_image_obj
        if split in {"test", "val"}:
            ann_images[split].append(ann_image_obj)
        # go through captions and add them to dict split
        captions = image['sentences']
        sentids = image['sentids']
        for sentid, caption in zip(sentids, captions):
            ann_cap_obj = {
                "id": sentid,
                "image_id": image_id,
                "caption": caption
            }

            raw_caption = caption['raw']
            caption_id = image_name[:-4] + '#' + str(sentid)
            # add info to dicts
            if split != 'restval':
                full_dict[split]['image_id'].append(image_id)
                full_dict[split]['image_name'].append(image_name)
                full_dict[split]['caption_id'].append(caption_id)
                full_dict[split]['caption'].append(raw_caption)
                # save ann_cap_obj
                ann_annotations[split].append(ann_cap_obj)
            else:
                # supplement training dataset with restval
                full_dict['train']['image_id'].append(image_id)
                full_dict['train']['image_name'].append(image_name)
                full_dict['train']['caption_id'].append(caption_id)
                full_dict['train']['caption'].append(raw_caption)
                # save ann_cap_obj
                ann_annotations['train'].append(ann_cap_obj)

    if not ann_path.is_dir():
        # if this is not a directory then make it, including every
        # parent that may be missing
        ann_path.mkdir(parents=True)

    for s in ["test", "val", "train"]:
        ann_dict[s]['images'] = ann_images[s]
        ann_dict[s]['annotations'] = ann_annotations[s]
        # save dict split as .json file
        with open(ann_path.joinpath(dataset + '_' + s + '.json'), 'w') \
                as ann_file:
            json.dump(ann_dict[s], ann_file)

    train_df = pd.DataFrame(train_dict, columns=columns)
    test_df = pd.DataFrame(test_dict, columns=columns)
    val_df = pd.DataFrame(val_dict, columns=columns)

    save_path = ROOT_PATH.joinpath('data', 'interim', 'karpathy_split')
    if not save_path.is_dir():
        # if dir does not exist create dir
        # create parents if they do not exist either
        save_path.mkdir(parents=True)

    train_file = save_path.joinpath(dataset + '_train.csv')
    test_file = save_path.joinpath(dataset + '_test.csv')
    val_file = save_path.joinpath(dataset + '_val.csv')

    train_df.to_csv(train_file)
    test_df.to_csv(test_file)
    val_df.to_csv(val_file)
    print("finished job!!")
