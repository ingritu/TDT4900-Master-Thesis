from pathlib import Path
from src.data.handle_karpathy_split import order_raw_data_and_move_to_interim
from src.data.text_to_csv import text_to_csv
from src.data.split_flickr8k import make_train_val_test_split

ROOT_PATH = Path(__file__).absolute().parents[2]


if __name__ == '__main__':
    # ####################### Karpathy Split ################## #
    datasets = ['coco', 'flickr8k', 'flickr30k']
    for dataset_ in datasets:
        data_path_ = ROOT_PATH.joinpath('data', 'raw', 'karpathy_split', 'dataset_' + dataset_ + '.json')
        order_raw_data_and_move_to_interim(data_path_, dataset_)

    # ######################## flickr8k ####################### #
    datasets = ['Flickr8k']
    for dataset_ in datasets:
        if dataset_ == 'Flickr8k':
            file_path_ = ROOT_PATH.joinpath('data', 'raw', dataset_,
                                            'Flickr_TextData',
                                            'Flickr8k.token.txt')
            save_path_ = ROOT_PATH.joinpath('data', 'interim', dataset_,
                                            'Flickr8k_token.csv')
        else:
            file_path_ = ROOT_PATH.joinpath('data', 'raw', dataset_,
                                            'results_20130124.token')
            save_path_ = ROOT_PATH.joinpath('data', 'interim', dataset_,
                                            'Flickr30k_token.csv')
        text_to_csv(file_path_, save_path_)

    # #################### Split Flickr8k ##################### #
    df_path_ = ROOT_PATH.joinpath('data', 'interim', 'Flickr8k',
                                  'Flickr8k_token.csv')
    train_path_ = ROOT_PATH.joinpath('data', 'raw', 'Flickr8k',
                                     'Flickr_TextData',
                                     'Flickr_8k.trainImages.txt')
    val_path_ = ROOT_PATH.joinpath('data', 'raw', 'Flickr8k',
                                   'Flickr_TextData',
                                   'Flickr_8k.devImages.txt')
    test_path_ = ROOT_PATH.joinpath('data', 'raw', 'Flickr8k',
                                    'Flickr_TextData',
                                    'Flickr_8k.testImages.txt')
    split_paths_ = [train_path_, val_path_, test_path_]
    save_path_ = ROOT_PATH.joinpath('data', 'interim', 'Flickr8k')
    make_train_val_test_split(df_path_, split_paths_, save_path_)
