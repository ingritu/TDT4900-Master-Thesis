from pathlib import Path
import pandas as pd
import json

ROOT_PATH = Path(__file__).absolute().parents[2]


def find_image_filename(images, image_id):
    for imgobj in images:
        if imgobj['id'] == image_id:
            return imgobj['file_name']
    return ''


def make_dataframe(data_path):
    with open(data_path, 'r') as json_file:
        data_dict = json.load(json_file)

    out_dict = {
        'image_id': [],
        'caption_id': [],
        'caption': []
    }
    images = data_dict['images']
    for capobj in data_dict['annotations']:
        im_id = find_image_filename(images,
                                    capobj['image_id'])
        cap_id = str(capobj['id'])
        caption = capobj['caption']
        out_dict['image_id'].append(im_id)
        out_dict['caption_id'].append(im_id + "#" + cap_id)
        out_dict['caption'].append(caption)

    data_df = pd.DataFrame(data=out_dict, columns=out_dict.keys())
    return data_df


def preprocess_coco(data_path, output_path, splits):
    for split in splits:
        d_path = data_path.joinpath('captions_' + split + '2014.json')
        split_df = make_dataframe(d_path)
        split_df.to_csv(output_path.joinpath('coco_' + split + '.csv'))


if __name__ == '__main__':
    dataset = "MSCOCO"
    data_path_ = ROOT_PATH.joinpath('data',
                                    'raw',
                                    dataset,
                                    'annotations')
    output_path_ = ROOT_PATH.joinpath('data',
                                      'interim',
                                      dataset)
    splits_ = ['train', 'val']
    preprocess_coco(data_path_, output_path_, splits_)



