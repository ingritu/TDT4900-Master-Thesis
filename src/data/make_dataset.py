from pathlib import Path
from src.data.handle_karpathy_split import order_raw_data_and_move_to_interim
from src.data.data_cleaning import basic_data_cleaning

ROOT_PATH = Path(__file__).absolute().parents[2]


if __name__ == '__main__':
    # ####################### Karpathy Split ################## #
    ann_path_ = ROOT_PATH.joinpath('data', 'processed', 'annotations')
    datasets = ['coco', 'flickr8k', 'flickr30k']
    for dataset_ in datasets:
        print("Preparing", dataset_)
        data_path_ = ROOT_PATH.joinpath('data', 'raw', 'karpathy_split',
                                        'dataset_' + dataset_ + '.json')
        order_raw_data_and_move_to_interim(data_path_, dataset_, ann_path_)

    # ################ DATA CLEANING ######################### #
    print("Cleaning Flickr8k!")
    df_path_ = ROOT_PATH.joinpath('data', 'interim', 'karpathy_split',
                                  'flickr8k_train.csv')
    save_path_ = ROOT_PATH.joinpath('data', 'interim', 'karpathy_split',
                                    'flickr8k_train_clean.csv')
    voc_save_path_ = ROOT_PATH.joinpath('data', 'interim', 'karpathy_split',
                                        'Flickr8k_vocabulary.csv')
    basic_data_cleaning(df_path_, save_path_, voc_save_path_)
