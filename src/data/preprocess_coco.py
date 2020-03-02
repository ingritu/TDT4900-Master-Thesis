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
        'image_name': [],
        'caption_id': [],
        'caption': []
    }

    images = data_dict['images']
    captions = data_dict['annotations']
    cap_counter = 0
    for imgobj in images:
        im_id = imgobj['id']
        im_name = imgobj['file_name']
        cap_ids, caps = find_captions(captions, imgobj['id'])
        for c in range(len(cap_ids)):
            cap_id = im_name + "#" + str(cap_ids[c])
            caption = caps[c]
            out_dict['image_id'].append(im_id)
            out_dict['image_name'].append(im_name)
            out_dict['caption_id'].append(im_name + "#" + cap_id)
            out_dict['caption'].append(caption)
            cap_counter += 1
            if cap_counter % 1000 == 0:
                print(cap_counter)

    data_df = pd.DataFrame(data=out_dict, columns=list(out_dict.keys()))
    return data_df


def preprocess_coco(data_path, output_path, splits):
    # TODO: modify to create annotation files too
    data_path = Path(data_path)
    output_path = Path(output_path)
    if not output_path.is_dir():
        # if not a dir then mkdir
        output_path.mkdir(parents=True)

    full_df = pd.DataFrame()
    for split in splits:
        d_path = data_path.joinpath('captions_' + split + '2014.json')
        split_df = make_dataframe(d_path)
        full_df = full_df.append(split_df, ignore_index=True)
        split_df.to_csv(output_path.joinpath('coco_' + split + '.csv'))
    full_df.to_csv(output_path.joinpath('coco_full.csv'))
