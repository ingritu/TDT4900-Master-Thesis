from pathlib import Path
import json


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


def sample_image_captions(models, seed):
    pass
