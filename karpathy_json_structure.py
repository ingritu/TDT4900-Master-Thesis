import json
from pathlib import Path

ROOT_PATH = Path(__file__).parent


def shallow_insights_about_dataset(data_path, dataset):
    with open(data_path, 'r') as json_file:
        data_dict = json.load(json_file)

    splitnames = {}
    filepath_names = {}
    images_with_more_than_five_captions = {'train': {}, 'test': {}, 'val': {}}

    data_images = data_dict['images']

    for image in data_images:
        image_split = image['split']
        if image_split not in splitnames.keys():
            splitnames[image_split] = 1
        else:
            splitnames[image_split] += 1
        if dataset == 'coco':
            image_filepath = image['filepath']
            if image_filepath not in filepath_names.keys():
                filepath_names[image_filepath] = 1
            else:
                filepath_names[image_filepath] += 1

        # find images with more than 5 captions
        num_caps = len(image['sentids'])
        if image_split != 'restval' and num_caps > 5:
            if num_caps not in images_with_more_than_five_captions[image_split].keys():
                images_with_more_than_five_captions[image_split][num_caps] = 1
            else:
                images_with_more_than_five_captions[image_split][num_caps] += 1
    print(splitnames)
    print(filepath_names)

    for key in images_with_more_than_five_captions.keys():
        print(key, images_with_more_than_five_captions[key])


if __name__ == '__main__':
    dataset_ = 'flickr30k'
    data_path_ = ROOT_PATH.joinpath('data', 'raw', 'karpathy_split', 'dataset_' + dataset_ + '.json')
    shallow_insights_about_dataset(data_path_, dataset_)
