from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT_PATH = Path(__file__).absolute().parents[2]


if __name__ == '__main__':
    # load cleaned training dataset
    coco_train_path = ROOT_PATH.joinpath('data',
                                         'interim',
                                         'karpathy_split',
                                         'coco_train_clean.csv')
    coco_train_df = pd.read_csv(coco_train_path)
    OG_SIZE = len(coco_train_df)

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
    REMOVE_IMG_SIZE = len(new_df)
    print('removed', OG_SIZE - REMOVE_IMG_SIZE, 'captions')
    print('few captions', few)
    # these numbers are dependent on how rigorous the initial cleaning was,
    # in particular the unk-percentage, which may remove captions.
    # removed 32 captions as a result of not enough caps
    # total removed caps are 344 unk=0.4
    # 432 unk=0.3

    save_path = ROOT_PATH.joinpath('data', 'interim', 'karpathy_split',
                                   'coco_sub_prep.csv')
    new_df.to_csv(save_path)
