from pathlib import Path
import pandas as pd
import json

ROOT_PATH = Path(__file__).absolute().parents[2]


def find_captions(captions, image_id):
    cap_ids = []
    caps = []
    remove_objects = []
    for capobj in captions:
        if capobj['image_id'] == image_id:
            cap_ids.append(capobj['id'])
            caps.append(capobj['caption'])
            remove_objects.append(capobj)
    # remove used captions
    for c in remove_objects:
        captions.remove(c)
    # print('cap_size', len(captions))
    return cap_ids, caps


def make_dataframe(data_path):
    with open(data_path, 'r') as json_file:
        data_dict = json.load(json_file)

    out_dict = {
        'image_id': [],
        'caption_id': [],
        'caption': []
    }

    images = data_dict['images']
    captions = data_dict['annotations']
    image_counter = 0
    for imgobj in images:
        im_id = imgobj['file_name']
        cap_ids, caps = find_captions(captions, imgobj['id'])
        for c in range(len(cap_ids)):
            cap_id = im_id + "#" + str(cap_ids[c])
            caption = captions[c]
            out_dict['image_id'].append(im_id)
            out_dict['caption_id'].append(im_id + "#" + cap_id)
            out_dict['caption'].append(caption)
            image_counter += 1
            if image_counter % 1000 == 0:
                print(image_counter)

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



