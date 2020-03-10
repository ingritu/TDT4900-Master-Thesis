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
    print("Started the make dataset script.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Dataset to train on. The options are '
                             '{flickr8k, flickr30k, coco}.')
    parser.add_argument('--karpathy', action='store_true',
                        help='Boolean used to decide whether to train on '
                             'the karpathy split of dataset or not.')
    parser.add_argument('--threshold', type=int, default=5,
                        help='Minimum word frequency for words included '
                             'in the vocabulary.')
    parser.add_argument('--unk-percentage', type=float, default=0.3,
                        help='The percentage of UNK tokens in a caption '
                             'must be below this value in order to be '
                             'included in the train set.')
    parser.add_argument('--cutoff-value', type=int, default=16,
                        help='As a part of the pre-processing we will '
                             'augment captions that are considered too '
                             'long. This argument essentially sets the '
                             'max length of a caption, excluding the '
                             'startseq and endseq tokens. The default '
                             'value is 16.')
    args = vars(parser.parse_args())

    # print all args
    print("using parsed arguments.")
    for key in args:
        print(key, args[key])

    # ####################### PRE-PROCESSING ################## #
    raw_path = ROOT_PATH.joinpath('data', 'raw')
    interim_path = ROOT_PATH.joinpath('data', 'interim')
    processed_path = ROOT_PATH.joinpath('data', 'processed')
    ann_path_ = processed_path.joinpath('annotations')
    dataset_ = args['dataset']
    assert dataset_ in {'flickr8k', 'flickr30k', 'coco'}, \
        dataset_ + " is not supported. Only flickr8k, flickr30k and coco " \
                   "are supported."
    print("Preparing", dataset_)
    if args['karpathy']:
        raw_path = raw_path.joinpath('karpathy_split')
        ann_path_ = ann_path_.joinpath('karpathy_split')
        interim_path = interim_path.joinpath('karpathy_split')
        data_path_ = raw_path.joinpath('dataset_' + dataset_ + '.json')
        order_raw_data_and_move_to_interim(data_path_, dataset_, ann_path_)
    elif dataset_ == 'coco':
        # preprocess coco
        data_path_ = raw_path.joinpath('MSCOCO', 'annotations')
        splits_ = ['train', 'val']
        preprocess_coco(data_path_, interim_path, splits_)
    else:
        print('Illegal Arguments. '
              'Only coco is supported if karpathy flag is not set.')
        exit(0)

    # ################ DATA CLEANING ######################### #
    print("Cleaning", dataset_)
    threshold = args['threshold']
    unk_percentage = args['unk_percentage']
    df_path_ = interim_path.joinpath(dataset_ + '_train.csv')
    save_path_ = interim_path.joinpath(dataset_ + '_train_clean.csv')
    voc_save_path_ = interim_path.joinpath(dataset_ + '_vocabulary.csv')
    basic_data_cleaning(df_path_, save_path_, voc_save_path_,
                        threshold=threshold, unk_percentage=unk_percentage)
