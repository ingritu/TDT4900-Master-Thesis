from pathlib import Path
from src.data.handle_karpathy_split import order_raw_data_and_move_to_interim
from src.data.preprocess_coco import preprocess_coco
from src.data.data_cleaning import basic_data_cleaning

import argparse

ROOT_PATH = Path(__file__).absolute().parents[2]


if __name__ == '__main__':
    """
    To run script in terminal:
    python3 -m src.data.make_dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Dataset to train on. The options are '
                             '{flickr8k, flickr30k, coco}.')
    parser.add_argument('--karpathy', type=bool, default=True,
                        help='Boolean used to decide whether to train on '
                             'the karpathy split of dataset or not.')
    args = vars(parser.parse_args())

    # ####################### PRE-PROCESSING ################## #
    raw_path = ROOT_PATH.joinpath('data', 'raw')
    interim_path = ROOT_PATH.joinpath('data', 'interim')
    processed_path = ROOT_PATH.joinpath('data', 'processed')
    ann_path_ = processed_path.joinpath('annotations')
    dataset_ = args['dataset']
    print("Preparing", dataset_)
    if args['karpathy']:
        raw_path = raw_path.joinpath('karpathy_split')
        ann_path_ = ann_path_.joinpath('karpathy_split')
        data_path_ = raw_path.joinpath('dataset_' + dataset_ + '.json')
        order_raw_data_and_move_to_interim(data_path_, dataset_, ann_path_)
    elif dataset_ == 'coco':
        # preprocess coco
        data_path_ = raw_path.joinpath('MSCOCO', 'annotations')
        output_path_ = interim_path.joinpath(dataset_)
        splits_ = ['train', 'val']
        preprocess_coco(data_path_, output_path_, splits_)

    # ################ DATA CLEANING ######################### #
    print("Cleaning", dataset_)
    df_path_ = interim_path.joinpath(dataset_ + '_train.csv')
    save_path_ = interim_path.joinpath(dataset_ + '_train_clean.csv')
    voc_save_path_ = interim_path.joinpath(dataset_ + '_vocabulary.csv')
    basic_data_cleaning(df_path_, save_path_, voc_save_path_)
